from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def test_frontend_has_required_ui() -> None:
    assert "<h1>Xirvo</h1>" in HTML
    assert "AI Assistant" in HTML
    assert 'id="chat-log"' in HTML
    assert 'id="message-input"' in HTML
    assert 'id="send-button"' in HTML
    assert 'id="new-conversation"' in HTML
    assert 'id="typing-indicator"' in HTML
    assert 'id="error-banner"' in HTML
    assert "/static/styles.css" in HTML
    assert "/static/app.js" in HTML


def test_styles_align_user_and_assistant_messages() -> None:
    assert ".message.user" in CSS
    assert "align-self: flex-end" in CSS
    assert ".message.assistant" in CSS
    assert "align-self: flex-start" in CSS


def test_frontend_session_chat_and_refresh_behavior() -> None:
    assert 'STORAGE_KEY = "xirvo_conversation_id"' in JS
    assert "localStorage" in JS
    assert "/api/chat/session" in JS
    assert "/api/chat" in JS
    assert "restoreConversation" in JS
    assert "startNewConversation" in JS
    assert "New Conversation" in HTML


def test_rapid_messages_are_batched_not_concatenated() -> None:
    assert "BATCH_DELAY_MS = 600" in JS
    assert "pendingMessages.push(text)" in JS
    assert "messages," in JS
    assert "messages.join" not in JS
    assert "pendingMessages.join" not in JS


def test_enter_sends_and_shift_enter_makes_newline() -> None:
    assert 'event.key === "Enter"' in JS
    assert "!event.shiftKey" in JS


def test_frontend_does_not_contain_secrets() -> None:
    combined = HTML + CSS + JS
    assert "GEMINI_API_KEY" not in combined
    assert "DB_PASSWORD" not in combined
    assert "postgresql" not in combined.lower()
    assert "INTERNAL_MIN_HOURLY_RATE" not in combined
    assert "hidden minimum" not in combined.lower()
