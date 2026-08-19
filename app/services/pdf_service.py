from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from app.services.exceptions import KnowledgeBaseError


def resolve_knowledge_base_pdf(path: Path, project_root: Path | None = None) -> Path:
    if path.is_file():
        return path

    data_dir = path.parent if path.parent.name else None
    if data_dir is None or not data_dir.is_dir():
        root = project_root or path.parent
        data_dir = root / "data"

    pdfs = sorted(
        candidate
        for candidate in data_dir.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    )
    if len(pdfs) == 1:
        return pdfs[0]

    available = ", ".join(pdf.name for pdf in pdfs) if pdfs else "(none)"
    raise KnowledgeBaseError(
        f"Knowledge-base PDF was not found at: {path}. "
        f"PDF files in {data_dir}: {available}"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False
    return "\n".join(cleaned).strip()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    if not path.exists():
        raise KnowledgeBaseError(f"Knowledge-base PDF was not found at: {path}")

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise KnowledgeBaseError(f"Unable to open the knowledge-base PDF: {exc}") from exc

    try:
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(document, start=1):
            extracted = clean_text(page.get_text("text") or "")
            if extracted:
                pages.append((index, extracted))
    finally:
        document.close()

    if not pages:
        raise KnowledgeBaseError(
            "The knowledge-base PDF contains no extractable text. "
            "A scanned/image-only PDF is not supported in this phase."
        )
    return pages
