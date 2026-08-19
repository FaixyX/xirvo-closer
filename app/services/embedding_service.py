from __future__ import annotations

from google import genai
from google.genai import types

from app.config import Settings
from app.services.exceptions import EmbeddingDimensionError


class EmbeddingService:
    def __init__(self, settings: Settings, client: genai.Client | None = None) -> None:
        self._settings = settings
        self._client = client or genai.Client(api_key=settings.gemini_api_key)

    async def embed_document(self, text: str) -> list[float]:
        embeddings = await self.embed_documents([text])
        return embeddings[0]

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed(text, task_type="RETRIEVAL_QUERY")

    async def embed_documents(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_many(batch, task_type="RETRIEVAL_DOCUMENT"))
        return vectors

    async def _embed(self, text: str, task_type: str) -> list[float]:
        vectors = await self._embed_many([text], task_type=task_type)
        return vectors[0]

    async def _embed_many(self, texts: list[str], task_type: str) -> list[list[float]]:
        response = await self._client.aio.models.embed_content(
            model=self._settings.gemini_embedding_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._settings.gemini_embedding_dimension,
            ),
        )
        vectors: list[list[float]] = []
        for embedding in response.embeddings or []:
            values = list(embedding.values or [])
            expected = self._settings.gemini_embedding_dimension
            if len(values) != expected:
                raise EmbeddingDimensionError(
                    f"Gemini returned {len(values)} embedding values; "
                    f"expected exactly {expected}. Embeddings were not truncated or resized."
                )
            vectors.append(values)
        if len(vectors) != len(texts):
            raise EmbeddingDimensionError(
                "Gemini did not return one embedding for every input text."
            )
        return vectors
