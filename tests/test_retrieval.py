import uuid

import pytest
from sqlalchemy import select

from app.db.models import Document, DocumentChunk
from app.services.rag_service import RAGService
from tests.conftest import FakeEmbeddingService, cleanup_document


def _vector(dimension: int, primary: float, secondary: float = 0.0) -> list[float]:
    values = [0.0] * dimension
    values[0] = primary
    values[1] = secondary
    return values


@pytest.mark.asyncio
async def test_cosine_retrieval_returns_relevant_chunks(db_session, settings) -> None:
    filename = f"retrieval-{uuid.uuid4()}.pdf"
    document = Document(filename=filename, content_hash=uuid.uuid4().hex)
    db_session.add(document)
    await db_session.flush()
    relevant = DocumentChunk(
        document_id=document.id,
        page_number=1,
        chunk_index=0,
        content="Xirvo develops custom AI chatbots for operations teams.",
        embedding=_vector(settings.gemini_embedding_dimension, 1.0, 0.1),
    )
    irrelevant = DocumentChunk(
        document_id=document.id,
        page_number=2,
        chunk_index=1,
        content="Office plants and kitchen snacks are restocked weekly.",
        embedding=_vector(settings.gemini_embedding_dimension, 0.0, 1.0),
    )
    db_session.add_all([relevant, irrelevant])
    await db_session.commit()

    class QueryEmbeddingService(FakeEmbeddingService):
        async def embed_query(self, text: str) -> list[float]:
            await super().embed_query(text)
            return _vector(settings.gemini_embedding_dimension, 1.0, 0.0)

    try:
        service = RAGService(settings, QueryEmbeddingService(settings.gemini_embedding_dimension))
        chunks = await service.retrieve(db_session, "Do you develop AI chatbots?")
        assert chunks
        assert chunks[0].content == relevant.content
        assert chunks[0].page_number == 1
        assert chunks[0].similarity >= chunks[-1].similarity
        stored = await db_session.scalar(select(DocumentChunk).where(DocumentChunk.id == relevant.id))
        assert stored is not None
        assert float(stored.embedding[0]) == 1.0
        assert float(stored.embedding[1]) == 0.1
    finally:
        await cleanup_document(db_session, filename)
