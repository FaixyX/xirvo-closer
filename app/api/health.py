from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_db)) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PostgreSQL is unavailable: {exc}",
        ) from exc

    try:
        result = await session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        enabled = result.scalar_one_or_none() is not None
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to check pgvector. Enable it with: "
                f"CREATE EXTENSION IF NOT EXISTS vector; ({exc})"
            ),
        ) from exc

    if not enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "pgvector is not enabled on this PostgreSQL database. "
                "Connect to the database and run: CREATE EXTENSION IF NOT EXISTS vector;"
            ),
        )

    return HealthResponse(status="ok", database="ok", pgvector="ok")
