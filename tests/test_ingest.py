import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.models import Document, DocumentChunk
from app.services.ingest_service import ingest_knowledge_base
from tests.conftest import FakeEmbeddingService, cleanup_document, write_pdf


def _pdf(tmp_path: Path, text: str) -> Path:
    marker = uuid.uuid4().hex
    return write_pdf(tmp_path / f"kb-{marker}.pdf", [f"{text} [{marker}]"])


@pytest.mark.asyncio
async def test_ingest_skips_unchanged_pdf(db_session, settings, tmp_path: Path) -> None:
    pdf_path = _pdf(tmp_path, "Xirvo builds AI chatbots and voice agents.")
    embeddings = FakeEmbeddingService(settings.gemini_embedding_dimension)
    first = await ingest_knowledge_base(db_session, settings, embeddings, pdf_path)
    first_calls = list(embeddings.document_calls)
    second = await ingest_knowledge_base(db_session, settings, embeddings, pdf_path)
    try:
        assert first.status == "ingested"
        assert first.chunk_count >= 1
        chunk = await db_session.scalar(
            select(DocumentChunk).join(Document).where(Document.filename == pdf_path.name)
        )
        assert chunk is not None
        assert len(list(chunk.embedding)) == 3072
        assert float(chunk.embedding[1]) == 4.0
        assert second.status == "skipped"
        assert embeddings.document_calls == first_calls
    finally:
        await cleanup_document(db_session, pdf_path.name)


@pytest.mark.asyncio
async def test_updated_pdf_is_reindexed(db_session, settings, tmp_path: Path) -> None:
    marker = uuid.uuid4().hex
    pdf_path = write_pdf(tmp_path / f"kb-{marker}.pdf", [f"Original Xirvo portfolio example: Acme analytics. [{marker}]"])
    embeddings = FakeEmbeddingService(settings.gemini_embedding_dimension)
    first = await ingest_knowledge_base(db_session, settings, embeddings, pdf_path)
    write_pdf(pdf_path, [f"Updated Xirvo portfolio example: Northwind logistics. [{marker}]"])
    second = await ingest_knowledge_base(db_session, settings, embeddings, pdf_path)
    try:
        assert first.status == "ingested"
        assert second.status == "ingested"
        assert first.content_hash != second.content_hash
        result = await db_session.execute(
            select(DocumentChunk.content).join(Document).where(Document.filename == pdf_path.name)
        )
        contents = " ".join(row[0] for row in result.all())
        assert "Northwind logistics" in contents
        assert "Acme analytics" not in contents
        documents = (
            await db_session.execute(select(Document).where(Document.filename == pdf_path.name))
        ).scalars().all()
        assert len(documents) == 1
    finally:
        await cleanup_document(db_session, pdf_path.name)


@pytest.mark.asyncio
async def test_failed_ingest_does_not_leave_partial_chunks(db_session, settings, tmp_path: Path) -> None:
    marker = uuid.uuid4().hex
    pdf_path = write_pdf(tmp_path / f"kb-{marker}.pdf", [f"Stable Xirvo company information. [{marker}]"])
    embeddings = FakeEmbeddingService(settings.gemini_embedding_dimension)
    await ingest_knowledge_base(db_session, settings, embeddings, pdf_path)
    original_hash = await db_session.scalar(
        select(Document.content_hash).where(Document.filename == pdf_path.name)
    )
    write_pdf(pdf_path, [f"Changed Xirvo company information that should not persist. [{marker}]"])

    async def fail_commit() -> None:
        raise RuntimeError("commit failed")

    original_commit = db_session.commit
    db_session.commit = fail_commit  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="commit failed"):
            await ingest_knowledge_base(db_session, settings, embeddings, pdf_path)
    finally:
        db_session.commit = original_commit  # type: ignore[method-assign]

    try:
        document = await db_session.scalar(select(Document).where(Document.filename == pdf_path.name))
        assert document is not None
        assert document.content_hash == original_hash
        result = await db_session.execute(
            select(DocumentChunk.content).where(DocumentChunk.document_id == document.id)
        )
        contents = " ".join(row[0] for row in result.all())
        assert "Stable Xirvo company information" in contents
        assert "should not persist" not in contents
    finally:
        await cleanup_document(db_session, pdf_path.name)
