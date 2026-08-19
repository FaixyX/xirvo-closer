from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import PROJECT_ROOT, Settings
from app.db.models import Document, DocumentChunk
from app.db.session import assert_pgvector_available
from app.services.chunking import chunk_pages
from app.services.embedding_service import EmbeddingService
from app.services.exceptions import KnowledgeBaseError
from app.services.pdf_service import extract_pages, hash_file, resolve_knowledge_base_pdf


@dataclass(frozen=True)
class IngestResult:
    status: str
    filename: str
    content_hash: str
    chunk_count: int
    reason: str | None = None


async def ingest_knowledge_base(
    session: AsyncSession,
    settings: Settings,
    embedding_service: EmbeddingService,
    pdf_path: Path | None = None,
) -> IngestResult:
    await assert_pgvector_available(session)

    path = resolve_knowledge_base_pdf(
        Path(pdf_path or settings.resolved_knowledge_base_path),
        PROJECT_ROOT,
    )

    content_hash = hash_file(path)
    filename = path.name

    existing_hash = await session.scalar(
        select(Document).where(Document.content_hash == content_hash)
    )
    if existing_hash is not None:
        chunk_count = await session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == existing_hash.id)
        )
        return IngestResult(
            status="skipped",
            filename=filename,
            content_hash=content_hash,
            chunk_count=int(chunk_count or 0),
            reason="PDF is unchanged; embeddings were not regenerated.",
        )

    pages = extract_pages(path)
    chunks = chunk_pages(
        pages,
        max_tokens=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise KnowledgeBaseError("No chunks could be generated from the knowledge-base PDF.")

    embeddings = await embedding_service.embed_documents([chunk.content for chunk in chunks])

    try:
        document = await session.scalar(select(Document).where(Document.filename == filename))
        if document is None:
            document = Document(filename=filename, content_hash=content_hash)
            session.add(document)
            await session.flush()
        else:
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
            )
            document.content_hash = content_hash
            await session.flush()

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    embedding=embedding,
                )
            )
        await session.commit()
    except Exception:
        await session.rollback()
        raise

    return IngestResult(
        status="ingested",
        filename=filename,
        content_hash=content_hash,
        chunk_count=len(chunks),
    )
