import io
import zipfile
from unittest.mock import patch
import pytest
from httpx import AsyncClient, ASGITransport
from tortoise import Tortoise

from app.main import app
from app.ingestion import (
    normalize_text,
    estimate_token_count,
    detect_source_format,
    extract_from_text,
    extract_from_pdf,
    extract_from_epub,
    parse_document,
    build_contextual_embedding_text,
    chunk_document,
    ParsedDocument,
    ParsedSection,
)
from app.cqrs import BookCommandHandler, CreateBookCommand, UpdateBookCommand
from app.models import Book, BookChunk, AuditLog


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
async def initialize_ingestion_tests():
    await Tortoise.init(
        db_url="sqlite://:memory:",
        modules={"models": ["app.models"]},
    )
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()


@pytest.mark.anyio
async def test_normalize_text_and_token_estimation():
    raw = "Line 1   \r\nLine 2\r\n\r\n\r\n\r\nLine 3\t  \n"
    cleaned = normalize_text(raw)
    assert cleaned == "Line 1\nLine 2\n\nLine 3"
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""

    assert estimate_token_count("") == 0
    short_tokens = estimate_token_count("Hello world!")
    long_tokens = estimate_token_count("Hello world! " * 20)
    assert short_tokens >= 3
    assert long_tokens > short_tokens * 10


@pytest.mark.anyio
async def test_detect_source_format_and_parse_document():
    assert detect_source_format(filename="book.PDF") == "pdf"
    assert detect_source_format(filename="novel.epub") == "epub"
    assert detect_source_format(filename="notes.md") == "markdown"
    assert detect_source_format(filename="readme.markdown") == "markdown"
    assert detect_source_format(filename="plain.txt") == "text"

    # Magic bytes detection when filename is absent or generic
    assert detect_source_format(file_bytes=b"%PDF-1.7 fake pdf content") == "pdf"
    assert detect_source_format(file_bytes=b"PK\x03\x04 fake epub zip") == "epub"
    assert detect_source_format(file_bytes=b"Just plain UTF-8 bytes") == "text"

    parsed_txt = parse_document(file_bytes=b"# Heading\nBody text", filename="doc.md")
    assert parsed_txt.source_format == "markdown"
    assert parsed_txt.sections[0].chapter == "Heading"


@pytest.mark.anyio
async def test_extract_from_text_chapters_and_markdown_headings():
    md_text = (
        "Preface remarks before any heading.\n\n"
        "# Chapter 1: Awakening\n"
        "The first morning in the citadel.\n\n"
        "## Part II: The Journey\n"
        "Crossing the desert plains.\n\n"
        "Epilogue: Return\n"
        "Final reflections at dusk."
    )
    doc = extract_from_text(md_text, source_format="markdown")
    assert doc.source_format == "markdown"
    assert len(doc.sections) == 4
    assert doc.sections[0].chapter is None
    assert "Preface remarks" in doc.sections[0].content
    assert doc.sections[1].chapter == "Chapter 1: Awakening"
    assert doc.sections[2].chapter == "Part II: The Journey"
    assert doc.sections[3].chapter == "Epilogue: Return"

    empty_doc = extract_from_text("   \n  ")
    assert empty_doc.content == ""
    assert empty_doc.sections == []


@pytest.mark.anyio
async def test_extract_from_pdf_pymupdf_and_pypdf_fallback():
    import fitz

    pdf_doc = fitz.open()
    pdf_doc.set_metadata({"title": "Orbital Mechanics", "author": "Ada Lovelace"})
    p1 = pdf_doc.new_page()
    p1.insert_text((72, 72), "Chapter 1: Hohmann Transfers\nElliptical orbits connect two circular orbits.")
    # Add an intentionally blank page to verify empty pages are skipped
    pdf_doc.new_page()
    p3 = pdf_doc.new_page()
    p3.insert_text((72, 72), "Chapter 2: Lagrange Points\nFive equilibrium points exist in a two-body system.")
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    parsed = extract_from_pdf(pdf_bytes)
    assert parsed.source_format == "pdf"
    assert parsed.title == "Orbital Mechanics"
    assert parsed.author == "Ada Lovelace"
    assert len(parsed.sections) == 2
    assert parsed.sections[0].page_number == 1
    assert parsed.sections[0].chapter == "Chapter 1: Hohmann Transfers"
    assert parsed.sections[1].page_number == 3
    assert parsed.sections[1].chapter == "Chapter 2: Lagrange Points"

    # Verify fallback to pypdf when fitz.open fails
    with patch("fitz.open", side_effect=RuntimeError("Simulated PyMuPDF failure")):
        fallback_parsed = extract_from_pdf(pdf_bytes)
        assert fallback_parsed.source_format == "pdf"
        assert len(fallback_parsed.sections) == 2
        assert "Hohmann Transfers" in fallback_parsed.content
        assert "Lagrange Points" in fallback_parsed.content


@pytest.mark.anyio
async def test_extract_from_epub_ebooklib_and_zip_fallback():
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("id-12345")
    book.set_title("Deep Sea Chronicles")
    book.set_language("en")
    book.add_author("Jacques Piccard")

    c1 = epub.EpubHtml(title="Chapter 1", file_name="chap_01.xhtml", lang="en")
    c1.content = (
        "<html><head><title>Ignored Title</title><script>evil()</script></head>"
        "<body><h1>Chapter 1: The Mariana Trench</h1>"
        "<p>Bathyscaphe Trieste descended into Challenger Deep.</p></body></html>"
    )
    c2 = epub.EpubHtml(title="Chapter 2", file_name="chap_02.xhtml", lang="en")
    c2.content = (
        "<html><body><nav>Skip Nav</nav><h2>Chapter 2: Bioluminescence</h2>"
        "<p>Anglerfish produce light through symbiotic bacteria.</p></body></html>"
    )
    book.add_item(c1)
    book.add_item(c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]

    buf = io.BytesIO()
    epub.write_epub(buf, book)
    epub_bytes = buf.getvalue()

    parsed = extract_from_epub(epub_bytes)
    assert parsed.source_format == "epub"
    assert parsed.title == "Deep Sea Chronicles"
    assert parsed.author == "Jacques Piccard"
    assert any("Mariana Trench" in s.content for s in parsed.sections)
    assert any("Bioluminescence" in s.content for s in parsed.sections)
    assert "evil()" not in parsed.content
    assert "Skip Nav" not in parsed.content

    # Verify ZIP container fallback when ebooklib.epub.read_epub raises
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("OEBPS/nav.xhtml", "<html><body><nav>Table of Contents</nav></body></html>")
        zf.writestr(
            "OEBPS/part1.xhtml",
            "<html><body><h1>Part 1: Tectonic Plates</h1><p>Subduction zones recycle oceanic crust.</p></body></html>",
        )
    with patch("ebooklib.epub.read_epub", side_effect=RuntimeError("Simulated ebooklib error")):
        zip_parsed = extract_from_epub(zip_buf.getvalue())
        assert zip_parsed.source_format == "epub"
        assert len(zip_parsed.sections) == 1
        assert zip_parsed.sections[0].chapter == "Part 1: Tectonic Plates"
        assert "Subduction zones" in zip_parsed.content


@pytest.mark.anyio
async def test_hierarchical_chunk_document_and_contextual_headers():
    header_text = build_contextual_embedding_text(
        chunk_text="Sample passage content.",
        title="Dune",
        author="Frank Herbert",
        chapter="Book One",
        page_number=42,
    )
    assert header_text.startswith("Book: Dune by Frank Herbert | Chapter: Book One | Page: 42\n")

    long_sec_1 = "Arrakis teaches the attitude of the knife—chopping off what's incomplete. " * 30
    long_sec_2 = "The spice melange extends life and enables foldspace navigation. " * 30
    parsed_doc = ParsedDocument(
        content=f"{long_sec_1}\n\n{long_sec_2}",
        source_format="text",
        sections=[
            ParsedSection(content=long_sec_1, chapter="Chapter 1: Caladan", page_number=1),
            ParsedSection(content=long_sec_2, chapter="Chapter 2: Arrakeen", page_number=15),
        ],
    )

    chunks = chunk_document(
        title="Dune",
        author="Frank Herbert",
        parsed_doc=parsed_doc,
        chunk_size_tokens=80,
        chunk_overlap_tokens=15,
    )
    assert len(chunks) >= 6
    # Verify sequential chunk indices
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Verify token bounds and contextual header injection
    for c in chunks:
        assert c.token_count <= 95
        assert "Book: Dune by Frank Herbert" in c.embedding_text
        assert c.chapter in ("Chapter 1: Caladan", "Chapter 2: Arrakeen")
        assert c.page_number in (1, 15)


@pytest.mark.anyio
async def test_ingest_uploaded_book_metadata_resolution_and_api_validation():
    # Test metadata fallback when title/author are omitted on upload
    md_bytes = (
        b"# Chapter 1: Neural Pathways\n"
        b"Synaptic plasticity strengthens connections between firing neurons."
    )
    book = await BookCommandHandler.ingest_uploaded_book(
        file_bytes=md_bytes,
        filename="synaptic_plasticity.md",
        title=None,
        author=None,
        user_id=None,
    )
    assert book.title == "synaptic_plasticity"
    assert book.author == "Unknown Author"
    assert book.source_format == "markdown"
    assert book.chunk_count == 1
    assert book.ingestion_status == "completed"

    # Test API upload validation errors (unsupported extension & empty file)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post("/users", json={"username": "ingest_tester", "password_hash": "pass123"})
        token_res = await ac.post("/token", data={"username": "ingest_tester", "password": "pass123"})
        headers = {"Authorization": f"Bearer {token_res.json()['access_token']}"}

        bad_ext_res = await ac.post(
            "/books/upload",
            files={"file": ("malware.exe", b"binary data", "application/octet-stream")},
            headers=headers,
        )
        assert bad_ext_res.status_code == 400
        assert "Unsupported file format" in bad_ext_res.json()["detail"]

        empty_res = await ac.post(
            "/books/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
            headers=headers,
        )
        assert empty_res.status_code == 400
        assert "empty" in empty_res.json()["detail"].lower()

        # Valid markdown upload via API
        ok_res = await ac.post(
            "/books/upload",
            files={"file": ("quantum_computing.md", md_bytes, "text/markdown")},
            data={"title": "Quantum Notes", "author": "Dr. Bell"},
            headers=headers,
        )
        assert ok_res.status_code == 200
        created = ok_res.json()
        assert created["title"] == "Quantum Notes"
        assert created["source_format"] == "markdown"
        assert created["chunk_count"] >= 1

        # Invalid UUID on /books/{id}/chunks returns 400
        bad_uuid_res = await ac.get("/books/not-a-uuid/chunks", headers=headers)
        assert bad_uuid_res.status_code == 400
