from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.db.models import LeadProfile, LLMUsage, Message, MessageRole, TechnicalFit, CommercialFit


class ConversationNotFoundError(LookupError):
    pass


async def _commit_or_flush(session: AsyncSession, commit: bool) -> None:
    if commit:
        await session.commit()
    else:
        await session.flush()


async def create_session(session: AsyncSession) -> uuid.UUID:
    conversation_id = uuid.uuid4()
    session.add_all(
        [
            Message(
                conversation_id=conversation_id,
                role=MessageRole.user.value,
                content={"messages": []},
            ),
            Message(
                conversation_id=conversation_id,
                role=MessageRole.assistant.value,
                content={"messages": []},
            ),
            LeadProfile(
                id=conversation_id,
                technical_fit=TechnicalFit.pending.value,
                commercial_fit=CommercialFit.pending.value,
            ),
        ]
    )
    await session.commit()
    return conversation_id


async def _lock_message_rows(session: AsyncSession, conversation_id: uuid.UUID) -> dict[str, Message]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .with_for_update()
    )
    rows = result.scalars().all()
    by_role = {row.role: row for row in rows}
    if MessageRole.user.value not in by_role or MessageRole.assistant.value not in by_role:
        raise ConversationNotFoundError(f"Conversation {conversation_id} was not found.")
    return by_role


def _message_list(row: Message) -> list[dict]:
    payload = row.content or {}
    messages = payload.get("messages") or []
    return list(messages)


def _max_sequence(rows: dict[str, Message]) -> int:
    sequences: list[int] = []
    for row in rows.values():
        for item in _message_list(row):
            sequences.append(int(item.get("sequence", 0)))
    return max(sequences) if sequences else 0


def _write_messages(row: Message, messages: list[dict]) -> None:
    row.content = {"messages": messages}
    flag_modified(row, "content")
    row.updated_at = datetime.now(timezone.utc)


async def append_user_messages(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    texts: list[str],
    commit: bool = True,
) -> dict:
    cleaned = [text.strip() for text in texts if text and text.strip()]
    if not cleaned:
        raise ValueError("At least one non-empty user message is required.")

    rows = await _lock_message_rows(session, conversation_id)
    user_row = rows[MessageRole.user.value]
    existing = _message_list(user_row)
    sequence = _max_sequence(rows)
    batch_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    appended: list[dict] = []
    for text in cleaned:
        sequence += 1
        entry = {
            "sequence": sequence,
            "batch_id": batch_id,
            "text": text,
            "created_at": created_at,
        }
        existing.append(entry)
        appended.append(entry)
    _write_messages(user_row, existing)
    await _commit_or_flush(session, commit)
    return {"batch_id": batch_id, "messages": appended}


async def append_assistant_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    text: str,
    batch_id: str,
    commit: bool = True,
) -> dict:
    rows = await _lock_message_rows(session, conversation_id)
    assistant_row = rows[MessageRole.assistant.value]
    existing = _message_list(assistant_row)
    sequence = _max_sequence(rows) + 1
    entry = {
        "sequence": sequence,
        "batch_id": batch_id,
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    existing.append(entry)
    _write_messages(assistant_row, existing)
    await _commit_or_flush(session, commit)
    return entry


async def get_history(session: AsyncSession, conversation_id: uuid.UUID) -> list[dict]:
    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id)
    )
    rows = result.scalars().all()
    if not rows:
        raise ConversationNotFoundError(f"Conversation {conversation_id} was not found.")

    merged: list[dict] = []
    for row in rows:
        for item in _message_list(row):
            merged.append(
                {
                    "sequence": int(item["sequence"]),
                    "role": row.role,
                    "text": item["text"],
                    "batch_id": item.get("batch_id"),
                    "created_at": item.get("created_at"),
                }
            )
    merged.sort(key=lambda item: item["sequence"])
    return merged


async def record_usage(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    commit: bool = True,
) -> LLMUsage:
    usage = LLMUsage(
        conversation_id=conversation_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    session.add(usage)
    await _commit_or_flush(session, commit)
    return usage


async def sum_usage(session: AsyncSession, conversation_id: uuid.UUID) -> dict[str, int]:
    result = await session.execute(
        select(LLMUsage).where(LLMUsage.conversation_id == conversation_id)
    )
    rows = result.scalars().all()
    return {
        "input_tokens": sum(row.input_tokens for row in rows),
        "output_tokens": sum(row.output_tokens for row in rows),
        "total_tokens": sum(row.total_tokens for row in rows),
    }


async def conversation_exists(session: AsyncSession, conversation_id: uuid.UUID) -> bool:
    result = await session.execute(
        select(Message.id).where(Message.conversation_id == conversation_id).limit(1)
    )
    return result.scalar_one_or_none() is not None
