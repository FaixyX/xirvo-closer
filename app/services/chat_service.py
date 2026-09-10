from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.exceptions import GeminiGenerationError
from app.services.gemini_service import GeminiService, SalesTurnResult
from app.services.lead_service import apply_fit_update, get_or_create_lead_profile
from app.services.message_service import (
    append_assistant_message,
    append_user_messages,
    get_history,
    record_usage,
)
from app.services.rag_service import RAGService
from app.tools.qualification_tool import EVALUATE_LEAD_FIT_NAME, run_evaluate_lead_fit


@dataclass(frozen=True)
class ChatTurnResult:
    conversation_id: UUID
    response: str
    handoff_required: bool
    next_action: str
    technical_fit: str
    commercial_fit: str


async def handle_chat(
    session: AsyncSession,
    conversation_id: UUID,
    user_texts: list[str],
    rag_service: RAGService,
    gemini_service: GeminiService,
) -> ChatTurnResult:
    appended = await append_user_messages(session, conversation_id, user_texts)
    question = "\n".join(user_texts)
    retrieved = await rag_service.retrieve(session, question)
    context_chunks = rag_service.to_prompt_chunks(retrieved)

    history = await get_history(session, conversation_id)
    previous = [item for item in history if item.get("batch_id") != appended["batch_id"]]
    profile = await get_or_create_lead_profile(session, conversation_id)

    async def execute_tool(name: str, arguments: dict) -> dict:
        if name != EVALUATE_LEAD_FIT_NAME:
            return {
                "technical_fit": profile.technical_fit,
                "commercial_fit": profile.commercial_fit,
                "public_pricing_guidance": "Unknown tool.",
                "human_handoff_recommended": False,
            }
        result = run_evaluate_lead_fit(
            arguments,
            current_technical_fit=profile.technical_fit,
            current_commercial_fit=profile.commercial_fit,
        )
        updated = await apply_fit_update(
            session,
            conversation_id,
            result["technical_fit"],
            result["commercial_fit"],
        )
        profile.technical_fit = updated.technical_fit
        profile.commercial_fit = updated.commercial_fit
        return result

    try:
        turn: SalesTurnResult = await gemini_service.generate_sales_turn(
            history=previous,
            context_chunks=context_chunks,
            user_texts=user_texts,
            lead_state={
                "technical_fit": profile.technical_fit,
                "commercial_fit": profile.commercial_fit,
            },
            tool_executor=execute_tool,
        )
    except GeminiGenerationError:
        raise
    except Exception as exc:
        raise GeminiGenerationError(str(exc)) from exc

    handoff_required = _handoff_required(turn, profile.technical_fit, profile.commercial_fit)

    await append_assistant_message(
        session,
        conversation_id,
        turn.assistant_response,
        appended["batch_id"],
        commit=False,
    )
    for usage in turn.usage_events:
        await record_usage(
            session,
            conversation_id,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            commit=False,
        )
    await session.commit()
    return ChatTurnResult(
        conversation_id=conversation_id,
        response=turn.assistant_response,
        handoff_required=handoff_required,
        next_action=turn.next_action,
        technical_fit=profile.technical_fit,
        commercial_fit=profile.commercial_fit,
    )


def _handoff_required(turn: SalesTurnResult, technical_fit: str, commercial_fit: str) -> bool:
    observations = turn.observations or {}
    if turn.handoff_required:
        return True
    if observations.get("asks_for_human") or observations.get("contract_or_nda_question"):
        return True
    if observations.get("ready_to_proceed") and technical_fit == "qualified":
        return True
    if technical_fit == "qualified" and commercial_fit == "negotiation_required":
        return True
    return False
