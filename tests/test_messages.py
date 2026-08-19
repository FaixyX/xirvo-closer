import asyncio

import pytest
from sqlalchemy import func, select

from app.db.models import Message
from app.db.session import AsyncSessionLocal
from app.services.message_service import (
    append_assistant_message,
    append_user_messages,
    create_session,
    get_history,
)
from tests.conftest import cleanup_conversation


@pytest.mark.asyncio
async def test_session_creates_exactly_two_message_rows(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        count = await db_session.scalar(
            select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
        )
        roles = (
            await db_session.execute(
                select(Message.role).where(Message.conversation_id == conversation_id)
            )
        ).scalars().all()
        assert count == 2
        assert set(roles) == {"user", "assistant"}
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_user_and_assistant_messages_append_with_global_sequence(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        appended = await append_user_messages(
            db_session,
            conversation_id,
            [
                "Do you develop AI chatbots?",
                "What about AI voice agents?",
                "Which technologies do you use?",
            ],
        )
        assistant = await append_assistant_message(
            db_session,
            conversation_id,
            "Xirvo develops custom AI chatbots and voice agents.",
            appended["batch_id"],
        )
        history = await get_history(db_session, conversation_id)
        assert [item["sequence"] for item in history] == [1, 2, 3, 4]
        assert [item["role"] for item in history] == ["user", "user", "user", "assistant"]
        assert history[0]["text"] == "Do you develop AI chatbots?"
        assert history[1]["text"] == "What about AI voice agents?"
        assert history[2]["text"] == "Which technologies do you use?"
        assert {item["batch_id"] for item in history[:3]} == {appended["batch_id"]}
        assert assistant["sequence"] == 4
        rows = (
            await db_session.execute(select(Message).where(Message.conversation_id == conversation_id))
        ).scalars().all()
        assert len(rows) == 2
        user_row = next(row for row in rows if row.role == "user")
        assert len(user_row.content["messages"]) == 3
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_history_reconstruction_sorts_by_sequence(db_session) -> None:
    conversation_id = await create_session(db_session)
    try:
        first = await append_user_messages(db_session, conversation_id, ["What services does Xirvo offer?"])
        await append_assistant_message(db_session, conversation_id, "Xirvo offers custom software delivery.", first["batch_id"])
        second = await append_user_messages(db_session, conversation_id, ["How much does it cost?"])
        await append_assistant_message(db_session, conversation_id, "That information is unavailable in the current context.", second["batch_id"])
        history = await get_history(db_session, conversation_id)
        assert [item["role"] for item in history] == ["user", "assistant", "user", "assistant"]
        assert history[2]["text"] == "How much does it cost?"
    finally:
        await cleanup_conversation(db_session, conversation_id)


@pytest.mark.asyncio
async def test_concurrent_user_updates_do_not_lose_messages(db_available) -> None:
    async with AsyncSessionLocal() as setup:
        conversation_id = await create_session(setup)

    async def write_message(text: str) -> None:
        async with AsyncSessionLocal() as session:
            await append_user_messages(session, conversation_id, [text])

    try:
        await asyncio.gather(
            write_message("Do you build chatbots?"),
            write_message("Do you build voice agents?"),
        )
        async with AsyncSessionLocal() as session:
            history = await get_history(session, conversation_id)
            texts = {item["text"] for item in history}
            sequences = [item["sequence"] for item in history]
            assert texts == {"Do you build chatbots?", "Do you build voice agents?"}
            assert sorted(sequences) == [1, 2]
            assert len(set(sequences)) == 2
            count = await session.scalar(
                select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
            )
            assert count == 2
    finally:
        async with AsyncSessionLocal() as session:
            await cleanup_conversation(session, conversation_id)
