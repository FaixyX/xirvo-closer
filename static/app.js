const STORAGE_KEY = "xirvo_conversation_id";
const BATCH_DELAY_MS = 600;

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const newConversationButton = document.getElementById("new-conversation");
const typingIndicator = document.getElementById("typing-indicator");
const errorBanner = document.getElementById("error-banner");

let conversationId = null;
let pendingMessages = [];
let batchTimer = null;
let sending = false;

function showError(message) {
  errorBanner.hidden = false;
  errorBanner.textContent = message;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.textContent = text;
  chatLog.appendChild(bubble);
  scrollToBottom();
}

function setTyping(visible) {
  typingIndicator.hidden = !visible;
}

function resizeComposer() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

async function createSession() {
  const data = await requestJson("/api/chat/session", { method: "POST" });
  conversationId = data.conversation_id;
  localStorage.setItem(STORAGE_KEY, conversationId);
  return conversationId;
}

async function startNewConversation() {
  clearError();
  pendingMessages = [];
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }
  chatLog.innerHTML = "";
  setTyping(false);
  await createSession();
}

async function restoreConversation() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (!saved) {
    await createSession();
    return;
  }
  try {
    const data = await requestJson(`/api/chat/${saved}`);
    conversationId = saved;
    chatLog.innerHTML = "";
    for (const message of data.messages || []) {
      appendMessage(message.role, message.text);
    }
  } catch (_error) {
    localStorage.removeItem(STORAGE_KEY);
    await createSession();
  }
}

async function sendBatch(messages) {
  if (!conversationId) {
    await createSession();
  }
  return requestJson("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      conversation_id: conversationId,
      messages,
    }),
  });
}

async function flushPending() {
  if (sending || pendingMessages.length === 0) {
    return;
  }
  sending = true;
  const batch = pendingMessages.splice(0, pendingMessages.length);
  clearError();
  setTyping(true);
  sendButton.disabled = true;
  try {
    const data = await sendBatch(batch);
    appendMessage("assistant", data.response);
  } catch (error) {
    showError(error.message || "The assistant could not answer right now.");
  } finally {
    setTyping(false);
    sendButton.disabled = false;
    sending = false;
    if (pendingMessages.length > 0) {
      batchTimer = setTimeout(flushPending, BATCH_DELAY_MS);
    }
  }
}

function queueMessage(text) {
  appendMessage("user", text);
  pendingMessages.push(text);
  if (batchTimer) {
    clearTimeout(batchTimer);
  }
  batchTimer = setTimeout(flushPending, BATCH_DELAY_MS);
}

function submitComposer() {
  const text = messageInput.value.trim();
  if (!text) {
    return;
  }
  messageInput.value = "";
  resizeComposer();
  queueMessage(text);
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitComposer();
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitComposer();
  }
});

messageInput.addEventListener("input", resizeComposer);
newConversationButton.addEventListener("click", () => {
  startNewConversation().catch((error) => {
    showError(error.message || "Unable to start a new conversation.");
  });
});

restoreConversation().catch((error) => {
  showError(error.message || "Unable to start a chat session.");
});
