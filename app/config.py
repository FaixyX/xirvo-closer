from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_database_urls(
    database_url: str | None = None,
    *,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "xirvo_rag",
    db_user: str = "postgres",
    db_password: str = "",
) -> tuple[URL, URL]:
    """Return (sync, async) SQLAlchemy URLs for psycopg.

    A full DATABASE_URL (Render) takes precedence over split DB_* fields.
    Render connections always use SSL. Local split-field connections do not.
    """
    if database_url:
        raw = database_url.strip()
        if raw.startswith("postgres://"):
            raw = "postgresql://" + raw[len("postgres://") :]
        parsed = make_url(raw)
        query = dict(parsed.query)
        if not any(key.lower() == "sslmode" for key in query):
            query["sslmode"] = "require"
        parsed = parsed.set(query=query)
        return (
            parsed.set(drivername="postgresql+psycopg"),
            parsed.set(drivername="postgresql+psycopg_async"),
        )

    sync_url = URL.create(
        "postgresql+psycopg",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name,
    )
    async_url = URL.create(
        "postgresql+psycopg_async",
        username=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        database=db_name,
    )
    return sync_url, async_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    gemini_api_key: str
    gemini_model: str = "gemini-3.7-flash"

    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_embedding_dimension: int = 3072

    database_url: str | None = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "xirvo_rag"
    db_user: str = "postgres"
    db_password: str = ""

    rag_top_k: int = 5
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 80

    knowledge_base_path: str = "data/Xirvo Sales Chatbot — RAG Knowledge Base.pdf"

    standard_hourly_rate: float = 40
    min_project_amount: float = 500
    internal_min_hourly_rate: float

    @field_validator("database_url", mode="before")
    @classmethod
    def _empty_database_url(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def resolved_knowledge_base_path(self) -> Path:
        path = Path(self.knowledge_base_path)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    def _engine_urls(self) -> tuple[URL, URL]:
        return build_database_urls(
            self.database_url,
            db_host=self.db_host,
            db_port=self.db_port,
            db_name=self.db_name,
            db_user=self.db_user,
            db_password=self.db_password,
        )

    @property
    def sync_database_url(self) -> URL:
        return self._engine_urls()[0]

    @property
    def async_database_url(self) -> URL:
        return self._engine_urls()[1]


@lru_cache
def get_settings() -> Settings:
    return Settings()
