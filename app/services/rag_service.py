from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import DocumentChunk
from app.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    page_number: int
    content: str
    similarity: float


class RAGService:
    def __init__(self, settings: Settings, embedding_service: EmbeddingService) -> None:
        self._settings = settings
        self._embedding_service = embedding_service

    async def retrieve(self, session: AsyncSession, question: str) -> list[RetrievedChunk]:
        query_embedding = await self._embedding_service.embed_query(question)
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.page_number,
                DocumentChunk.content,
                (1 - distance).label("similarity"),
            )
            .order_by(distance)
            .limit(self._settings.rag_top_k)
        )
        result = await session.execute(stmt)
        rows = result.all()
        return [
            RetrievedChunk(
                id=str(row.id),
                page_number=row.page_number,
                content=row.content,
                similarity=float(row.similarity or 0.0),
            )
            for row in rows
        ]

    @staticmethod
    def to_prompt_chunks(chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {"page_number": chunk.page_number, "content": chunk.content}
            for chunk in chunks
        ]
