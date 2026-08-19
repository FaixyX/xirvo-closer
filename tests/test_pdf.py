from pathlib import Path

from app.services.exceptions import KnowledgeBaseError
from app.services.pdf_service import clean_text, extract_pages, hash_file, resolve_knowledge_base_pdf
from tests.conftest import write_pdf


def test_extract_pages_preserves_page_numbers(tmp_path: Path) -> None:
    pdf_path = write_pdf(
        tmp_path / "kb.pdf",
        [
            "Xirvo builds custom software on page one.",
            "Page two describes AI voice agents.",
        ],
    )
    pages = extract_pages(pdf_path)
    by_number = {page_number: text for page_number, text in pages}
    assert 1 in by_number
    assert 2 in by_number
    assert "page one" in by_number[1].lower()
    assert "voice agents" in by_number[2].lower()


def test_extract_pages_rejects_empty_pdf(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "empty.pdf", [""])
    try:
        extract_pages(pdf_path)
        raise AssertionError("Expected KnowledgeBaseError")
    except KnowledgeBaseError as exc:
        assert "no extractable text" in str(exc).lower()


def test_missing_pdf_raises_clear_error(tmp_path: Path) -> None:
    try:
        extract_pages(tmp_path / "missing.pdf")
        raise AssertionError("Expected KnowledgeBaseError")
    except KnowledgeBaseError as exc:
        assert "not found" in str(exc).lower()


def test_hash_file_is_sha256(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "hash.pdf", ["Stable knowledge base text."])
    digest = hash_file(pdf_path)
    assert len(digest) == 64
    assert digest == hash_file(pdf_path)


def test_clean_text_collapses_whitespace() -> None:
    cleaned = clean_text("Xirvo   offers\n\n\nAI   chatbots")
    assert "Xirvo offers" in cleaned
    assert "\n\n\n" not in cleaned


def test_resolve_knowledge_base_pdf_uses_only_pdf_in_folder(tmp_path: Path) -> None:
    pdf_path = write_pdf(tmp_path / "Xirvo Sales Chatbot — RAG Knowledge Base.pdf", ["Xirvo knowledge."])
    resolved = resolve_knowledge_base_pdf(tmp_path / "xirvo_knowledge_base.pdf")
    assert resolved == pdf_path
