from collections.abc import AsyncGenerator

from pgvector.psycopg import register_vector, register_vector_async
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.compat import configure_windows_event_loop
from app.config import get_settings
from app.services.exceptions import PgVectorNotAvailableError

configure_windows_event_loop()

_settings = get_settings()

_engine_kwargs = {
    "pool_pre_ping": True,
    "pool_size": 5,
    "max_overflow": 5,
    "pool_recycle": 300,
}

async_engine = create_async_engine(
    _settings.async_database_url,
    **_engine_kwargs,
)
sync_engine = create_engine(
    _settings.sync_database_url,
    **_engine_kwargs,
)


def _register_vector(dbapi_connection, _connection_record) -> None:
    # SQLAlchemy's async psycopg dialect wraps the real driver connection.
    if hasattr(dbapi_connection, "run_async"):
        dbapi_connection.run_async(register_vector_async)
        return
    register_vector(dbapi_connection)


event.listen(async_engine.sync_engine, "connect", _register_vector)
event.listen(sync_engine, "connect", _register_vector)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def assert_pgvector_available(session: AsyncSession) -> None:
    result = await session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    if result.scalar_one_or_none() is None:
        raise PgVectorNotAvailableError(
            "pgvector is not enabled on this PostgreSQL database. "
            "Connect to the database and run: CREATE EXTENSION IF NOT EXISTS vector;"
        )
