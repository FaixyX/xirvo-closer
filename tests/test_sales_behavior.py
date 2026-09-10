import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.db.models import LeadProfile, Message
from app.services.chat_service import handle_chat
from app.services.gemini_service import GenerationResult, SalesTurnResult
from app.services.message_service import create_session, get_history
from tests.conftest import FakeRAGService, cleanup_conversation

ROOT = Path(__file__).resolve().parents[1]
SALES_PROMPT = (ROOT / "app" / "prompts" / "sales_closer_system.txt").read_text(encoding="utf-8")
RAG_PROMPT = (ROOT / "app" / "prompts" / "rag_system.txt").read_text(encoding="utf-8")
FRONTEND = (
    (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    + (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    + (ROOT / "static" / "styles.css").read_text(encoding="utf-8")
)


class SequenceGeminiService:
    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.index = 0
        self.calls: list[dict] = []

    async def generate_sales_turn(
        self,
        history,
        context_chunks,
        user_texts,
        lead_state=None,
        tool_executor=None,
    ) -> SalesTurnResult:
        spec = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        self.calls.append(
            {
                "history": history,
                "context_chunks": context_chunks,
                "user_texts": user_texts,
                "lead_state": lead_state,
            }
        )
        tool_executed = False
        if spec.get("tool_args") and tool_executor is not None:
            await tool_executor("evaluate_lead_fit", spec["tool_args"])
            tool_executed = True
        usages = spec.get("usage_events")
        if usages is None:
            usages = [
                GenerationResult(
                    text=spec["assistant_response"],
                    input_tokens=spec.get("input_tokens", 100),
                    output_tokens=spec.get("output_tokens", 25),
                    total_tokens=spec.get("total_tokens", 125),
                )
            ]
        return SalesTurnResult(
            assistant_response=spec["assistant_response"],
            next_action=spec.get("next_action", "continue_discovery"),
            observations=spec.get("observations", {}),
            qualification_tool_needed=spec.get("qualification_tool_needed", bool(spec.get("tool_args"))),
            handoff_required=spec.get("handoff_required", False),
            usage_events=usages,
            tool_executed=tool_executed,
        )


def test_sales_prompt_forbids_proactive_pricing() -> None:
    text = SALES_PROMPT.lower()
    assert "never proactively mention pricing" in text
    assert "technical qualification does not give you permission to introduce pricing" in text
    assert "until the customer initiates the commercial discussion" in text
    assert "how much would something like this cost?" in text


def test_sales_prompt_defines_consultative_closer_behavior() -> None:
    text = SALES_PROMPT.lower()
    assert "sales closer" in text
    assert "acknowledge" in text
    assert "do not interrogate" in text
    assert "standard development rate is $40" in text
    assert "minimum project engagement: $500" in text
    assert "weekly billing" in text
    assert "no free trial" in text
    assert "evaluate_lead_fit" in text
    assert "human handoff" in text
    assert "never invent xirvo facts" in text
    assert "native android" in text
    assert "native ios" in text
    assert "desktop" in text


def test_confidential_threshold_is_not_in_prompts_or_frontend(settings) -> None:
    confidential = str(int(settings.internal_min_hourly_rate))
    combined = SALES_PROMPT + RAG_PROMPT + FRONTEND
    assert f"${confidential}" not in combined
    assert "INTERNAL_MIN_HOURLY_RATE" not in combined
    assert "hidden minimum" not in combined.lower()
    assert "internal min" not in combined.lower()


def test_sales_prompt_avoids_implementation_onboarding_questions() -> None:
    text = SALES_PROMPT.lower()
    assert "do not ask implementation or onboarding questions" in text
    assert "api keys" in text
    assert "oauth credentials" in text
    assert "ssh access" in text
    assert "what should be built and why it creates value" in text
    assert "arranged during project onboarding" in text


def test_sales_prompt_does_not_guarantee_timelines() -> None:
    text = SALES_PROMPT.lower()
    assert "never guarantee that a project can be completed within a specific timeframe" in text
    assert "one week is certainly doable" in text
    assert "the team would need to review" in text
    assert "exact scope before confirming" in text


def test_minimum_rate_question_is_anchored_to_standard_rate() -> None:
    assert "What is your minimum rate?" in SALES_PROMPT
    assert "Xirvo's standard development rate is $40 per hour" in SALES_PROMPT
    assert "Never disclose any other internal figure" in SALES_PROMPT


@pytest.mark.asyncio
async def test_new_session_creates_pending_lead_profile(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        profile = await db_session.get(LeadProfile, conversation_id)
        assert profile is not None
        assert profile.technical_fit == "pending"
        assert profile.commercial_fit == "pending"
        message_count = await db_session.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
        )
        assert message_count == 2
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_existing_conversation_without_profile_is_backfilled(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        profile = await db_session.get(LeadProfile, conversation_id)
        await db_session.delete(profile)
        await db_session.commit()
        rag = FakeRAGService()
        gemini = SequenceGeminiService(
            [{"assistant_response": "Happy to help with a Xirvo software project. What are you looking to build?"}]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["We want a TypeScript SaaS platform."],
            rag,
            gemini,
        )
        created = await db_session.get(LeadProfile, conversation_id)
        assert created is not None
        assert created.technical_fit in {"pending", "qualified"}
        assert result.response
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_saas_lead_qualifies_technically(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        rag = FakeRAGService(
            chunks=[{"page_number": 3, "content": "Xirvo builds SaaS platforms and AI systems with TypeScript and Python."}]
        )
        gemini = SequenceGeminiService(
            [
                {
                    "assistant_response": (
                        "A TypeScript SaaS platform is a strong fit for Xirvo. "
                        "What is the main problem you want the product to solve first?"
                    ),
                    "next_action": "understand_problem",
                    "tool_args": {
                        "software_service_needed": True,
                        "project_type": "saas_web_application",
                        "technologies": ["typescript"],
                        "hourly_budget": None,
                        "project_budget": None,
                        "pricing_negotiation_requested": False,
                    },
                }
            ]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["We want a TypeScript SaaS platform for client onboarding."],
            rag,
            gemini,
        )
        assert result.technical_fit == "qualified"
        assert result.commercial_fit == "pending"
        assert "SaaS" in result.response
        assert rag.questions
        lowered = result.response.lower()
        assert "$40" not in result.response
        assert "$500" not in result.response
        assert "hour" not in lowered
        assert "billing" not in lowered
        assert "deposit" not in lowered
        assert "discount" not in lowered
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_android_lead_is_politely_disqualified(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        gemini = SequenceGeminiService(
            [
                {
                    "assistant_response": (
                        "Thanks for clarifying. Xirvo currently focuses on web, AI and automation "
                        "work with Python, JavaScript and TypeScript, so a native Android app "
                        "would not be a fit. I can still point you toward the kinds of software "
                        "services we do take on if that is useful."
                    ),
                    "next_action": "politely_disqualify",
                    "tool_args": {
                        "software_service_needed": True,
                        "project_type": "native_android",
                        "technologies": ["kotlin"],
                    },
                }
            ]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["We need a native Android application in Kotlin."],
            FakeRAGService(),
            gemini,
        )
        assert result.technical_fit == "unqualified"
        assert "native Android" in result.response
        assert "Python" in result.response
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_price_asked_immediately_does_not_negotiate(db_session, settings) -> None:
    conversation_id = await create_session(db_session)
    try:
        gemini = SequenceGeminiService(
            [
                {
                    "assistant_response": (
                        "Xirvo's standard development rate is $40 per hour. Before getting into "
                        "project-specific commercial discussions, I'd first like to understand "
                        "what you're looking to build so we can determine whether we're the right fit."
                    ),
                    "next_action": "understand_problem",
                    "tool_args": {
                        "software_service_needed": True,
                        "hourly_budget": 30,
                        "pricing_negotiation_requested": True,
                    },
                    "observations": {"hourly_budget": 30, "pricing_negotiation_requested": True},
                }
            ]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["Can you do $30/hour?"],
            FakeRAGService(),
            gemini,
        )
        confidential = str(int(settings.internal_min_hourly_rate))
        assert result.commercial_fit == "pending"
        assert "$40" in result.response
        assert confidential not in result.response
        assert f"${confidential}" not in result.response
        assert "what you're looking to build" in result.response
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_qualified_lead_negotiation_sets_handoff(db_session, settings) -> None:
    conversation_id = await create_session(db_session)
    try:
        first = SequenceGeminiService(
            [
                {
                    "assistant_response": "A TypeScript SaaS platform fits Xirvo's web capabilities. What should the first workflow do?",
                    "next_action": "understand_requirements",
                    "tool_args": {
                        "software_service_needed": True,
                        "project_type": "saas_web_application",
                        "technologies": ["typescript"],
                    },
                }
            ]
        )
        await handle_chat(
            db_session,
            conversation_id,
            ["We need a TypeScript SaaS platform."],
            FakeRAGService(),
            first,
        )
        second = SequenceGeminiService(
            [
                {
                    "assistant_response": (
                        "Xirvo's standard rate is $40 per hour, with a focus on reliable, secure, "
                        "maintainable and production-ready software. Since the project appears to "
                        "fit Xirvo's capabilities, project-specific pricing can be discussed with "
                        "the Xirvo team."
                    ),
                    "next_action": "human_handoff",
                    "handoff_required": True,
                    "observations": {"pricing_negotiation_requested": True, "price_objection": True},
                    "tool_args": {
                        "software_service_needed": True,
                        "project_type": "saas_web_application",
                        "technologies": ["typescript"],
                        "hourly_budget": 30,
                        "pricing_negotiation_requested": True,
                    },
                }
            ]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["$40 is a little high. Can you do better?"],
            FakeRAGService(),
            second,
        )
        confidential = str(int(settings.internal_min_hourly_rate))
        assert result.technical_fit == "qualified"
        assert result.commercial_fit == "negotiation_required"
        assert result.handoff_required is True
        assert "$40" in result.response
        assert "Xirvo team" in result.response
        assert confidential not in result.response
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_competitor_rate_objection_keeps_value_framing(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        gemini = SequenceGeminiService(
            [
                {
                    "assistant_response": (
                        "A lower hourly rate can look attractive initially, but the overall cost of "
                        "software also depends heavily on reliability, maintainability, security and "
                        "how much rework is required later. Xirvo's focus is on building software that "
                        "is ready to operate reliably in production."
                    ),
                    "next_action": "handle_objection",
                }
            ]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["Another agency quoted us $20/hour."],
            FakeRAGService(),
            gemini,
        )
        lowered = result.response.lower()
        assert "reliability" in lowered
        assert "cheap" not in lowered
        assert "worse" not in lowered
        assert "incompetent" not in lowered
        assert "$40" not in result.response or "Xirvo" in result.response
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_trust_objection_uses_rag_portfolio(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        rag = FakeRAGService(
            chunks=[
                {
                    "page_number": 12,
                    "content": "Orange Media Lab: Xirvo built CRM and AI qualification workflows.",
                }
            ]
        )
        gemini = SequenceGeminiService(
            [
                {
                    "assistant_response": (
                        "A relevant example is Orange Media Lab, where Xirvo built CRM and AI "
                        "qualification workflows. That same pattern of production-ready automation "
                        "is what we would apply to a similar requirement."
                    ),
                    "next_action": "show_relevant_portfolio",
                }
            ]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["How do I know you can deliver?"],
            rag,
            gemini,
        )
        assert rag.questions
        assert "Orange Media Lab" in result.response
        assert gemini.calls[0]["context_chunks"][0]["content"].startswith("Orange Media Lab")
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_buying_intent_stops_discovery_and_sets_handoff(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        await handle_chat(
            db_session,
            conversation_id,
            ["We need a TypeScript SaaS platform."],
            FakeRAGService(),
            SequenceGeminiService(
                [
                    {
                        "assistant_response": "That fits Xirvo's SaaS work. What is the first workflow you want live?",
                        "tool_args": {
                            "software_service_needed": True,
                            "project_type": "saas_web_application",
                            "technologies": ["typescript"],
                        },
                    }
                ]
            ),
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["That sounds good. Let's move forward."],
            FakeRAGService(),
            SequenceGeminiService(
                [
                    {
                        "assistant_response": (
                            "Great — the Xirvo team can continue from here. If you share your name "
                            "and email, they can pick up this conversation and outline the next step."
                        ),
                        "next_action": "human_handoff",
                        "handoff_required": True,
                        "observations": {"ready_to_proceed": True, "name_provided": False, "email_provided": False},
                    }
                ]
            ),
        )
        assert result.handoff_required is True
        assert result.technical_fit == "qualified"
        assert "Xirvo team" in result.response
        assert "timeline" not in result.response.lower()
        assert "budget" not in result.response.lower()
        assert "$40" not in result.response
        assert "$500" not in result.response
        assert "billing" not in result.response.lower()
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_existing_details_remain_in_history_and_are_not_reasked(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        first = SequenceGeminiService(
            [
                {
                    "assistant_response": "Thanks, Priya. Since HubSpot is already in place, we can look at automating qualification rather than replacing the CRM. Which fields should a lead satisfy before it is marked qualified?",
                    "next_action": "understand_requirements",
                    "tool_args": {
                        "software_service_needed": True,
                        "project_type": "crm",
                        "technologies": ["hubspot"],
                    },
                    "observations": {
                        "name_provided": True,
                        "email_provided": True,
                        "software_service_needed": True,
                        "project_type": "crm",
                    },
                }
            ]
        )
        await handle_chat(
            db_session,
            conversation_id,
            [
                "I'm Priya, email priya@example.com. Budget is $40/hour. We need HubSpot lead qualification automation."
            ],
            FakeRAGService(),
            first,
        )
        second = SequenceGeminiService(
            [{"assistant_response": "Got it — we can keep HubSpot and automate the qualification step. What does 'qualified' mean for your team today?"}]
        )
        result = await handle_chat(
            db_session,
            conversation_id,
            ["Qualified means they asked for a demo."],
            FakeRAGService(),
            second,
        )
        history = await get_history(db_session, conversation_id)
        joined = " ".join(item["text"] for item in history)
        assert "Priya" in joined
        assert "priya@example.com" in joined
        assert "$40/hour" in joined
        assert "name" not in result.response.lower()
        assert "email" not in result.response.lower()
        previous = " ".join(item["text"] for item in second.calls[0]["history"])
        assert "Priya" in previous
        assert "priya@example.com" in previous
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_multiple_gemini_calls_store_each_usage_event(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        gemini = SequenceGeminiService(
            [
                {
                    "assistant_response": "A TypeScript SaaS platform is within Xirvo's capabilities.",
                    "tool_args": {
                        "software_service_needed": True,
                        "project_type": "saas_web_application",
                        "technologies": ["typescript"],
                    },
                    "usage_events": [
                        GenerationResult(text="tool-call", input_tokens=40, output_tokens=8, total_tokens=48),
                        GenerationResult(text="final", input_tokens=55, output_tokens=12, total_tokens=67),
                    ],
                }
            ]
        )
        await handle_chat(
            db_session,
            conversation_id,
            ["We want a TypeScript SaaS platform."],
            FakeRAGService(),
            gemini,
        )
        from app.services.message_service import sum_usage

        usage = await sum_usage(db_session, conversation_id)
        assert usage["input_tokens"] == 95
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 115
    finally:
        await cleanup_conversation(db_session, conversation_id)


def test_api_chat_response_schema_stays_customer_facing() -> None:
    payload = json.dumps({"conversation_id": "x", "response": "hello", "handoff_required": True})
    from app.schemas import ChatResponse

    dumped = ChatResponse(conversation_id="11111111-1111-1111-1111-111111111111", response="hello").model_dump()
    assert set(dumped) == {"conversation_id", "response"}
    assert "handoff_required" not in dumped
    assert "technical_fit" not in dumped
    assert payload  # keep json import used for explicit serialization check
    assert "handoff_required" not in ChatResponse.model_json_schema()["properties"]
