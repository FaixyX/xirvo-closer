import math
from types import SimpleNamespace

import pytest

from app.services.embedding_service import EmbeddingService
from app.services.exceptions import EmbeddingDimensionError


class FakeEmbedResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = [SimpleNamespace(values=vector) for vector in vectors]


class FakeModels:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[dict] = []

    async def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeEmbedResponse(self.vectors)


class FakeClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.aio = SimpleNamespace(models=FakeModels(vectors))


def _raw_vector(dimension: int) -> list[float]:
    values = [0.0] * dimension
    values[0] = 3.0
    values[1] = 4.0
    return values


@pytest.mark.asyncio
async def test_embed_document_keeps_full_unnormalized_vector(settings) -> None:
    raw = _raw_vector(settings.gemini_embedding_dimension)
    client = FakeClient([raw])
    service = EmbeddingService(settings, client=client)

    result = await service.embed_document("Xirvo knowledge chunk")

    assert len(result) == 3072
    assert len(result) == settings.gemini_embedding_dimension
    assert result == raw
    assert result[0] == 3.0
    assert result[1] == 4.0
    assert math.isclose(math.sqrt(result[0] ** 2 + result[1] ** 2), 5.0)
    call = client.aio.models.calls[0]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == settings.gemini_embedding_dimension
    assert call["model"] == settings.gemini_embedding_model


@pytest.mark.asyncio
async def test_embed_query_uses_retrieval_query_task(settings) -> None:
    raw = _raw_vector(settings.gemini_embedding_dimension)
    client = FakeClient([raw])
    service = EmbeddingService(settings, client=client)

    result = await service.embed_query("What services does Xirvo offer?")

    assert result == raw
    assert client.aio.models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_wrong_dimension_is_rejected(settings) -> None:
    client = FakeClient([[0.1, 0.2, 0.3]])
    service = EmbeddingService(settings, client=client)
    with pytest.raises(EmbeddingDimensionError):
        await service.embed_document("bad vector")
