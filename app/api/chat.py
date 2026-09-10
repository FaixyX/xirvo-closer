from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_gemini_service, get_rag_service
from app.db.session import get_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryMessage,
    HistoryResponse,
    SessionResponse,
    UsageResponse,
)
from app.services.chat_service import handle_chat
from app.services.exceptions import GeminiGenerationError
from app.services.gemini_service import GeminiService
from app.services.message_service import (
    ConversationNotFoundError,
    conversation_exists,
    create_session,
    get_history,
    sum_usage,
)
from app.services.rag_service import RAGService

router = APIRouter()


@router.post("/chat/session", response_model=SessionResponse)
async def create_chat_session(session: AsyncSession = Depends(get_db)) -> SessionResponse:
    conversation_id = await create_session(session)
    return SessionResponse(conversation_id=conversation_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    session: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
) -> ChatResponse:
    if not await conversation_exists(session, payload.conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")

    user_texts = [text.strip() for text in payload.messages if text and text.strip()]
    if not user_texts:
        raise HTTPException(status_code=422, detail="At least one non-empty message is required.")

    try:
        result = await handle_chat(
            session,
            payload.conversation_id,
            user_texts,
            rag_service,
            gemini_service,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeminiGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini did not return a response: {exc}",
        ) from exc

    return ChatResponse(conversation_id=result.conversation_id, response=result.response)


@router.get("/chat/{conversation_id}", response_model=HistoryResponse)
async def load_history(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> HistoryResponse:
    try:
        history = await get_history(session, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return HistoryResponse(
        conversation_id=conversation_id,
        messages=[
            HistoryMessage(sequence=item["sequence"], role=item["role"], text=item["text"])
            for item in history
        ],
    )


@router.get("/chat/{conversation_id}/usage", response_model=UsageResponse)
async def load_usage(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> UsageResponse:
    if not await conversation_exists(session, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    totals = await sum_usage(session, conversation_id)
    return UsageResponse(**totals)
