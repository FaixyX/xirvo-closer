from functools import lru_cache

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService
from app.services.rag_service import RAGService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings())


@lru_cache
def get_gemini_service() -> GeminiService:
    return GeminiService(get_settings())


@lru_cache
def get_rag_service() -> RAGService:
    return RAGService(get_settings(), get_embedding_service())
