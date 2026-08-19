"""Ingest the Xirvo knowledge-base PDF into PostgreSQL + pgvector."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.compat import configure_windows_event_loop
from app.config import get_settings
from app.db.session import AsyncSessionLocal
from app.services.embedding_service import EmbeddingService
from app.services.exceptions import KnowledgeBaseError, PgVectorNotAvailableError
from app.services.ingest_service import ingest_knowledge_base


async def _run() -> int:
    settings = get_settings()
    embedding_service = EmbeddingService(settings)
    async with AsyncSessionLocal() as session:
        try:
            result = await ingest_knowledge_base(session, settings, embedding_service)
        except (KnowledgeBaseError, RuntimeError, PgVectorNotAvailableError) as exc:
            print(f"Ingestion failed: {exc}")
            return 1
        except Exception as exc:
            print(f"Ingestion failed: {exc}")
            return 1

    if result.status == "skipped":
        print(
            f"Skipped ingestion for {result.filename} "
            f"(hash {result.content_hash}). {result.reason}"
        )
        print(f"Existing chunks: {result.chunk_count}")
        return 0

    print(
        f"Ingested {result.filename} "
        f"({result.chunk_count} chunks, hash {result.content_hash})."
    )
    return 0


def main() -> None:
    configure_windows_event_loop()
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
