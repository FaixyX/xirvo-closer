from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str
    app_port: int
    debug: bool

    gemini_api_key: str
    gemini_model: str

    gemini_embedding_model: str
    gemini_embedding_dimension: int

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    rag_top_k: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    knowledge_base_path: str

    standard_hourly_rate: float
    min_project_amount: float
    internal_min_hourly_rate: float

    @property
    def resolved_knowledge_base_path(self) -> Path:
        path = Path(self.knowledge_base_path)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    def _database_url(self, drivername: str) -> URL:
        return URL.create(
            drivername=drivername,
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

    @property
    def sync_database_url(self) -> URL:
        return self._database_url("postgresql+psycopg")

    @property
    def async_database_url(self) -> URL:
        return self._database_url("postgresql+psycopg_async")


@lru_cache
def get_settings() -> Settings:
    return Settings()
