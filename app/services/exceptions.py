class KnowledgeBaseError(Exception):
    """Raised when the knowledge-base PDF cannot be used."""


class PgVectorNotAvailableError(Exception):
    """Raised when the pgvector extension is missing."""


class EmbeddingDimensionError(Exception):
    """Raised when Gemini returns an embedding of the wrong size."""


class GeminiGenerationError(Exception):
    """Raised when Gemini does not return a usable response."""
