from app.services.chunking import chunk_page, chunk_pages, count_tokens


def test_faq_question_and_answer_stay_together() -> None:
    text = (
        "Frequently Asked Questions\n\n"
        "Do you develop AI chatbots?\n"
        "Yes. Xirvo designs and develops custom AI chatbots for business workflows.\n\n"
        "What about AI voice agents?\n"
        "Xirvo also builds AI voice agents for inbound and outbound calls."
    )
    chunks = chunk_page(text, page_number=3, max_tokens=80, overlap=10)
    assert chunks
    assert all(chunk.page_number == 3 for chunk in chunks)
    combined = "\n".join(chunk.content for chunk in chunks)
    assert "Do you develop AI chatbots?" in combined
    assert "custom AI chatbots" in combined
    faq_chunk = next(chunk for chunk in chunks if "Do you develop AI chatbots?" in chunk.content)
    assert "custom AI chatbots" in faq_chunk.content


def test_large_section_splits_instead_of_dropping_text() -> None:
    sentence = "Xirvo delivers production-grade software for growing companies."
    text = " ".join([sentence] * 40)
    chunks = chunk_page(text, page_number=1, max_tokens=30, overlap=5)
    assert len(chunks) > 1
    assert "production-grade software" in chunks[0].content
    reconstructed = " ".join(chunk.content for chunk in chunks)
    assert "growing companies" in reconstructed


def test_overlap_keeps_adjacent_context() -> None:
    paragraphs = [f"Service description paragraph number {index} with extra detail." for index in range(1, 9)]
    chunks = chunk_page("\n\n".join(paragraphs), page_number=1, max_tokens=25, overlap=8)
    assert len(chunks) >= 2
    first_tail = chunks[0].content.split()[-6:]
    assert any(token in chunks[1].content for token in first_tail)


def test_chunk_pages_assigns_global_chunk_index() -> None:
    pages = [
        (1, "Company information. Xirvo is a software studio."),
        (2, "Pricing sections list discovery, build, and retainers."),
    ]
    chunks = chunk_pages(pages, max_tokens=40, overlap=5)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_number for chunk in chunks} == {1, 2}


def test_count_tokens_ignores_blank_text() -> None:
    assert count_tokens("   ") == 0
    assert count_tokens("Xirvo builds chatbots") > 0
