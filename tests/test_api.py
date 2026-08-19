import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_gemini_service, get_rag_service
from app.main import app
from tests.conftest import FakeGeminiService, FakeRAGService, cleanup_conversation


@pytest.fixture
def override_llm():
    rag = FakeRAGService(
        chunks=[{"page_number": 1, "content": "Xirvo develops custom AI chatbots and voice agents."}]
    )
    gemini = FakeGeminiService("Xirvo develops custom AI chatbots and voice agents.")
    app.dependency_overrides[get_rag_service] = lambda: rag
    app.dependency_overrides[get_gemini_service] = lambda: gemini
    yield rag, gemini
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_index_serves_frontend() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        home = await client.get("/")
        script = await client.get("/static/app.js")
    assert home.status_code == 200
    assert "Xirvo" in home.text
    assert "AI Assistant" in home.text
    assert script.status_code == 200
    assert "BATCH_DELAY_MS = 600" in script.text


@pytest.mark.asyncio
async def test_health_endpoint(db_available) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["pgvector"] == "ok"


@pytest.mark.asyncio
async def test_chat_session_history_and_usage(db_available, override_llm) -> None:
    _rag, gemini = override_llm
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_response = await client.post("/api/chat/session")
        assert session_response.status_code == 200
        conversation_id = session_response.json()["conversation_id"]
        try:
            chat_response = await client.post(
                "/api/chat",
                json={
                    "conversation_id": conversation_id,
                    "messages": [
                        "Do you develop AI chatbots?",
                        "What about AI voice agents?",
                    ],
                },
            )
            assert chat_response.status_code == 200
            body = chat_response.json()
            assert body["conversation_id"] == conversation_id
            assert "Xirvo" in body["response"]

            history_response = await client.get(f"/api/chat/{conversation_id}")
            assert history_response.status_code == 200
            messages = history_response.json()["messages"]
            assert [item["role"] for item in messages] == ["user", "user", "assistant"]
            assert messages[0]["text"] == "Do you develop AI chatbots?"
            assert messages[1]["text"] == "What about AI voice agents?"

            usage_response = await client.get(f"/api/chat/{conversation_id}/usage")
            assert usage_response.status_code == 200
            usage = usage_response.json()
            assert usage["input_tokens"] == 100
            assert usage["output_tokens"] == 25
            assert usage["total_tokens"] == 125

            follow_up = await client.post(
                "/api/chat",
                json={"conversation_id": conversation_id, "messages": ["Can you explain the second one?"]},
            )
            assert follow_up.status_code == 200
            assert gemini.calls[-1]["user_texts"] == ["Can you explain the second one?"]
            previous_texts = [item["text"] for item in gemini.calls[-1]["history"]]
            assert "Do you develop AI chatbots?" in previous_texts
            assert "What about AI voice agents?" in previous_texts
            usage_after = (await client.get(f"/api/chat/{conversation_id}/usage")).json()
            assert usage_after["input_tokens"] == 200
            assert usage_after["output_tokens"] == 50
            assert usage_after["total_tokens"] == 250
        finally:
            await cleanup_conversation(db_available, conversation_id)


@pytest.mark.asyncio
async def test_chat_unknown_conversation_returns_404(db_available, override_llm) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "conversation_id": "11111111-1111-1111-1111-111111111111",
                "messages": ["Hello"],
            },
        )
    assert response.status_code == 404
