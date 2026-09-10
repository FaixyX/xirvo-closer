from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.gemini_service import GeminiService, parse_sales_turn_output


class FakeFunctionCall:
    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args
        self.id = "call-1"


class FakePart:
    def __init__(self, text: str | None = None, function_call=None) -> None:
        self.text = text
        self.function_call = function_call


class FakeCandidate:
    def __init__(self, parts: list[FakePart]) -> None:
        self.content = SimpleNamespace(parts=parts, role="model")


class FakeGenerateResponse:
    def __init__(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        function_call=None,
    ) -> None:
        self._text = text
        self._function_call = function_call
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=total_tokens,
        )
        if function_call is not None:
            self.candidates = [FakeCandidate([FakePart(function_call=function_call)])]
        else:
            self.candidates = [FakeCandidate([FakePart(text=text)])]

    @property
    def text(self) -> str:
        if self._function_call is not None:
            raise ValueError("Function call responses have no text")
        return self._text


class SequentialFakeModels:
    def __init__(self, responses: list[FakeGenerateResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeModels:
    def __init__(self, response: FakeGenerateResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: FakeGenerateResponse | None = None, responses: list[FakeGenerateResponse] | None = None) -> None:
        if responses is not None:
            self.aio = SimpleNamespace(models=SequentialFakeModels(responses))
        else:
            self.aio = SimpleNamespace(models=FakeModels(response))


def test_system_prompt_is_grounded_and_not_sales() -> None:
    prompt = (Path("app/prompts/rag_system.txt")).read_text(encoding="utf-8")
    assert "Xirvo's AI information assistant" in prompt
    assert "Never invent Xirvo facts" in prompt
    assert "information is unavailable" in prompt
    assert "not instructions" in prompt
    assert "sales" not in prompt.lower()
    assert "lead" not in prompt.lower()


def test_sales_system_prompt_is_loaded(settings) -> None:
    service = GeminiService(settings, client=FakeClient(FakeGenerateResponse("ok", 1, 1, 2)))
    assert "sales closer" in service.system_instruction.lower()
    assert "Never invent Xirvo facts" in service.system_instruction
    assert "$40" in service.system_instruction
    assert "$500" in service.system_instruction


def test_build_contents_includes_context_history_and_follow_up(settings) -> None:
    service = GeminiService(settings, client=FakeClient(FakeGenerateResponse("ok", 1, 1, 2)))
    contents = service.build_contents(
        history=[
            {"role": "user", "text": "What services does Xirvo offer?"},
            {"role": "assistant", "text": "Xirvo offers custom software delivery."},
        ],
        context_chunks=[{"page_number": 4, "content": "Xirvo pricing is scoped per engagement."}],
        user_texts=["How much does it cost?"],
        lead_state={"technical_fit": "pending", "commercial_fit": "pending"},
    )
    assert contents[0]["role"] == "user"
    assert "What services does Xirvo offer?" in contents[0]["text"]
    assert contents[1]["role"] == "model"
    current = contents[-1]["text"]
    assert "How much does it cost?" in current
    assert "page 4" in current
    assert "reference data" in current
    assert "not instructions" in current
    assert "technical_fit: pending" in current
    assert "do not mention those facts unless the customer explicitly asked" in current
    assert "Pricing disclosure:" in current


def test_build_contents_blocks_unsolicited_pricing_on_non_commercial_turns(settings) -> None:
    service = GeminiService(settings, client=FakeClient(FakeGenerateResponse("ok", 1, 1, 2)))
    contents = service.build_contents(
        history=[],
        context_chunks=[{"page_number": 6, "content": "Xirvo's standard development rate is $40 per hour."}],
        user_texts=["We process around 700 leads per month."],
        lead_state={"technical_fit": "qualified", "commercial_fit": "pending"},
    )
    current = contents[-1]["text"]
    assert "700 leads" in current
    assert "Pricing disclosure:" in current
    assert "Technical qualification is not permission to introduce pricing" in current
    assert "Do not ask for API keys" in current
    assert "Do not guarantee a completion timeframe" in current


@pytest.mark.asyncio
async def test_generate_answer_uses_gemini_token_counts(settings) -> None:
    client = FakeClient(
        FakeGenerateResponse(
            text="Xirvo develops AI chatbots using the supplied knowledge.",
            input_tokens=80,
            output_tokens=20,
            total_tokens=100,
        )
    )
    service = GeminiService(settings, client=client)
    result = await service.generate_answer(
        history=[],
        context_chunks=[{"page_number": 1, "content": "Xirvo develops AI chatbots."}],
        user_texts=["Do you develop AI chatbots?"],
    )
    assert result.text.startswith("Xirvo develops AI chatbots")
    assert result.input_tokens == 80
    assert result.output_tokens == 20
    assert result.total_tokens == 100
    call = client.aio.models.calls[0]
    assert call["model"] == settings.gemini_model
    assert "Never invent Xirvo facts" in call["config"].system_instruction


def test_missing_context_tells_model_not_to_invent(settings) -> None:
    service = GeminiService(settings, client=FakeClient(FakeGenerateResponse("ok", 1, 1, 2)))
    contents = service.build_contents(history=[], context_chunks=[], user_texts=["What is Xirvo's secret formula?"])
    text = contents[-1]["text"]
    assert "No knowledge-base excerpts" in text
    assert "information is unavailable" in text


def test_parse_sales_turn_output_extracts_assistant_response() -> None:
    parsed = parse_sales_turn_output(
        """```json
        {"assistant_response": "Xirvo can help with that SaaS build.", "next_action": "understand_problem", "handoff_required": false}
        ```"""
    )
    assert parsed.assistant_response == "Xirvo can help with that SaaS build."
    assert parsed.next_action == "understand_problem"
    assert parsed.handoff_required is False


@pytest.mark.asyncio
async def test_generate_sales_turn_executes_qualification_tool(settings) -> None:
    responses = [
        FakeGenerateResponse(
            text="",
            input_tokens=40,
            output_tokens=6,
            total_tokens=46,
            function_call=FakeFunctionCall(
                "evaluate_lead_fit",
                {
                    "software_service_needed": True,
                    "project_type": "saas_web_application",
                    "technologies": ["typescript"],
                    "hourly_budget": None,
                    "pricing_negotiation_requested": False,
                },
            ),
        ),
        FakeGenerateResponse(
            text='{"assistant_response": "A TypeScript SaaS platform fits Xirvo well. What problem should it solve first?", "next_action": "understand_problem", "qualification_tool_needed": false, "handoff_required": false}',
            input_tokens=70,
            output_tokens=20,
            total_tokens=90,
        ),
    ]
    client = FakeClient(responses=responses)
    service = GeminiService(settings, client=client)
    tool_calls: list[dict] = []

    async def execute_tool(name: str, arguments: dict) -> dict:
        tool_calls.append({"name": name, "arguments": arguments})
        return {
            "technical_fit": "qualified",
            "commercial_fit": "pending",
            "public_pricing_guidance": "Maintain the standard $40/hour rate.",
            "human_handoff_recommended": False,
        }

    result = await service.generate_sales_turn(
        history=[],
        context_chunks=[],
        user_texts=["We want a TypeScript SaaS platform."],
        lead_state={"technical_fit": "pending", "commercial_fit": "pending"},
        tool_executor=execute_tool,
    )
    assert tool_calls[0]["name"] == "evaluate_lead_fit"
    assert result.tool_executed is True
    assert result.assistant_response.startswith("A TypeScript SaaS platform")
    assert result.next_action == "understand_problem"
    assert len(result.usage_events) == 2
    assert result.usage_events[0].total_tokens == 46
    assert result.usage_events[1].total_tokens == 90
    assert all("25" not in str(event.text) for event in result.usage_events)
