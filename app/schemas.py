from uuid import UUID

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    conversation_id: UUID


class ChatRequest(BaseModel):
    conversation_id: UUID
    messages: list[str] = Field(min_length=1)


class ChatResponse(BaseModel):
    conversation_id: UUID
    response: str


class HistoryMessage(BaseModel):
    sequence: int
    role: str
    text: str


class HistoryResponse(BaseModel):
    conversation_id: UUID
    messages: list[HistoryMessage]


class UsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class HealthResponse(BaseModel):
    status: str
    database: str
    pgvector: str
