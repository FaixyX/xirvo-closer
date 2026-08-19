from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,6}\s+\S.+"
    r"|[A-Z0-9][A-Z0-9 \-/&,:]{2,80}"
    r"|\d{1,3}[.\)]\s+\S.+"
    r"|[A-Z][A-Za-z0-9 \-/&]{0,70}"
    r")$"
)
_QUESTION_RE = re.compile(
    r"^(?:q(?:uestion)?\s*[:.)-]\s*|faq\s*[:.)-]\s*).+\??$|.+\?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    content: str


def count_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(_TOKEN_RE.findall(text))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if not parts:
            parts.append(token)
        elif re.fullmatch(r"[^\w\s]", token):
            parts.append(token)
        else:
            parts.append(" " + token)
    return "".join(parts)


def tail_tokens(text: str, n: int) -> str:
    if n <= 0:
        return ""
    tokens = tokenize(text)
    if not tokens:
        return ""
    return detokenize(tokens[-n:])


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if stripped.endswith((".", "!", "?")) and not stripped.endswith("?"):
        return False
    return bool(_HEADING_RE.match(stripped))


def _is_question(text: str) -> bool:
    first_line = text.strip().split("\n", 1)[0]
    return bool(_QUESTION_RE.match(first_line.strip()))


def split_into_sections(text: str) -> list[str]:
    lines = text.split("\n")
    sections: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if _is_heading(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    return ["\n".join(block).strip() for block in sections if "\n".join(block).strip()]


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [part.strip() for part in parts if part.strip()]


def merge_faq_pairs(paragraphs: list[str], max_tokens: int) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(paragraphs):
        current = paragraphs[index]
        if _is_question(current) and index + 1 < len(paragraphs):
            combined = f"{current}\n{paragraphs[index + 1]}"
            if count_tokens(combined) <= max_tokens:
                merged.append(combined)
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def split_sentences(text: str) -> list[str]:
    pieces = _SENTENCE_RE.split(text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    if max_tokens <= 0:
        return [text]
    return [
        detokenize(tokens[index : index + max_tokens])
        for index in range(0, len(tokens), max_tokens)
    ]


def split_to_fit(text: str, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]

    fitted: list[str] = []
    for sentence in split_sentences(text):
        if count_tokens(sentence) <= max_tokens:
            fitted.append(sentence)
        else:
            fitted.extend(hard_split(sentence, max_tokens))
    return fitted or hard_split(text, max_tokens)


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part.strip()).strip()


def _pack_units(units: list[str], page_number: int, max_tokens: int, overlap: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    overlap_prefix = ""

    def emit(content: str) -> None:
        nonlocal overlap_prefix
        if overlap_prefix and not content.startswith(overlap_prefix):
            content = f"{overlap_prefix}\n\n{content}".strip()
        if not content:
            return
        chunks.append(
            TextChunk(
                page_number=page_number,
                chunk_index=len(chunks),
                content=content,
            )
        )
        overlap_prefix = tail_tokens(content, overlap)

    def flush() -> None:
        nonlocal current_parts, current_tokens
        content = _join_parts(current_parts)
        current_parts = []
        current_tokens = 0
        emit(content)

    for unit in units:
        pieces = split_to_fit(unit, max_tokens) if count_tokens(unit) > max_tokens else [unit]
        for piece in pieces:
            piece_tokens = count_tokens(piece)
            if current_parts and current_tokens + piece_tokens > max_tokens:
                flush()
            current_parts.append(piece)
            current_tokens += piece_tokens

    if current_parts:
        flush()

    return chunks


def chunk_page(text: str, page_number: int, max_tokens: int, overlap: int) -> list[TextChunk]:
    if not text.strip():
        return []

    units: list[str] = []
    for section in split_into_sections(text):
        paragraphs = merge_faq_pairs(split_paragraphs(section), max_tokens)
        units.extend(paragraphs)

    if not units:
        units = [text.strip()]

    return _pack_units(units, page_number, max_tokens, overlap)


def chunk_pages(
    pages: list[tuple[int, str]],
    max_tokens: int,
    overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_number, text in pages:
        page_chunks = chunk_page(text, page_number, max_tokens, overlap)
        for item in page_chunks:
            chunks.append(
                TextChunk(
                    page_number=item.page_number,
                    chunk_index=len(chunks),
                    content=item.content,
                )
            )
    return chunks
