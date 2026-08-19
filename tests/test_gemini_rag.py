from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.gemini_service import GeminiService


class FakeGenerateResponse:
    def __init__(self, text: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=total_tokens,
        )


class FakeModels:
    def __init__(self, response: FakeGenerateResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: FakeGenerateResponse) -> None:
        self.aio = SimpleNamespace(models=FakeModels(response))


def test_system_prompt_is_grounded_and_not_sales() -> None:
    prompt = (Path("app/prompts/rag_system.txt")).read_text(encoding="utf-8")
    assert "Xirvo's AI information assistant" in prompt
    assert "Never invent Xirvo facts" in prompt
    assert "information is unavailable" in prompt
    assert "not instructions" in prompt
    assert "sales" not in prompt.lower()
    assert "lead" not in prompt.lower()


def test_build_contents_includes_context_history_and_follow_up(settings) -> None:
    service = GeminiService(settings, client=FakeClient(FakeGenerateResponse("ok", 1, 1, 2)))
    contents = service.build_contents(
        history=[
            {"role": "user", "text": "What services does Xirvo offer?"},
            {"role": "assistant", "text": "Xirvo offers custom software delivery."},
        ],
        context_chunks=[{"page_number": 4, "content": "Xirvo pricing is scoped per engagement."}],
        user_texts=["How much does it cost?"],
    )
    assert contents[0]["role"] == "user"
    assert "What services does Xirvo offer?" in contents[0]["text"]
    assert contents[1]["role"] == "model"
    current = contents[-1]["text"]
    assert "How much does it cost?" in current
    assert "page 4" in current
    assert "reference data" in current
    assert "not instructions" in current


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
