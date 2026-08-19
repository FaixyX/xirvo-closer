from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.compat import configure_windows_event_loop
from app.config import get_settings
from app.db.models import Document, DocumentChunk, LLMUsage, Message
from app.db.session import AsyncSessionLocal
from app.services.gemini_service import GenerationResult

configure_windows_event_loop()


class FakeEmbeddingService:
    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or get_settings().gemini_embedding_dimension
        self.document_calls: list[str] = []
        self.query_calls: list[str] = []

    def _vector_for(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        values[0] = float(len(text) + 3)
        values[1] = 4.0
        return values

    async def embed_document(self, text: str) -> list[float]:
        self.document_calls.append(text)
        return self._vector_for(text)

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector_for(text)

    async def embed_documents(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        self.document_calls.extend(texts)
        return [self._vector_for(text) for text in texts]


class FakeRAGService:
    def __init__(self, chunks: list[dict] | None = None) -> None:
        self.chunks = chunks or []
        self.questions: list[str] = []

    async def retrieve(self, session: AsyncSession, question: str):
        self.questions.append(question)
        return self.chunks

    @staticmethod
    def to_prompt_chunks(chunks) -> list[dict]:
        if chunks and isinstance(chunks[0], dict):
            return chunks
        return [{"page_number": chunk.page_number, "content": chunk.content} for chunk in chunks]


class FakeGeminiService:
    def __init__(self, text: str = "Xirvo answers from the knowledge base.") -> None:
        self.text = text
        self.calls: list[dict] = []

    async def generate_answer(self, history, context_chunks, user_texts) -> GenerationResult:
        self.calls.append(
            {
                "history": history,
                "context_chunks": context_chunks,
                "user_texts": user_texts,
            }
        )
        return GenerationResult(
            text=self.text,
            input_tokens=100,
            output_tokens=25,
            total_tokens=125,
        )


def write_pdf(path: Path, pages: list[str]) -> Path:
    document = pymupdf.open()
    for page_text in pages:
        page = document.new_page()
        if page_text:
            page.insert_textbox(page.rect, page_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return path


@pytest.fixture
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            extension = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            if extension.scalar_one_or_none() is None:
                pytest.skip("pgvector is not enabled")
            yield session
    except pytest.skip.Exception:
        raise
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not available: {exc}")


@pytest.fixture
async def db_available(db_session: AsyncSession) -> AsyncSession:
    return db_session


async def cleanup_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await session.execute(delete(LLMUsage).where(LLMUsage.conversation_id == conversation_id))
    await session.commit()


async def cleanup_document(session: AsyncSession, filename: str) -> None:
    document = await session.scalar(select(Document).where(Document.filename == filename))
    if document is not None:
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        await session.execute(delete(Document).where(Document.id == document.id))
        await session.commit()
