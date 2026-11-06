from dataclasses import dataclass, field
import io
import os
import re
import unicodedata
import zipfile
from typing import List, Optional, Callable

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "200"))
DEFAULT_CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "30"))

CHAPTER_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+(.+)|((?:chapter|part|book|prologue|epilogue)\b[^\n]{0,80}))$",
    re.IGNORECASE,
)


@dataclass
class ParsedSection:
    content: str
    chapter: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class ParsedDocument:
    content: str
    source_format: str = "text"
    title: Optional[str] = None
    author: Optional[str] = None
    sections: List[ParsedSection] = field(default_factory=list)


@dataclass
class ChunkPayload:
    chunk_index: int
    content: str
    embedding_text: str
    chapter: Optional[str] = None
    page_number: Optional[int] = None
    token_count: int = 0


def estimate_token_count(text: str) -> int:
    """Fast, deterministic word-piece token count estimator (~1.3 tokens per word/punctuation)."""
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]+", text, re.UNICODE)
    return max(1, int(len(tokens) * 1.15))


def normalize_text(text: str) -> str:
    """Normalize Unicode, strip control characters, and collapse excessive blank lines and dot leaders."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse table-of-contents dot leaders and long horizontal rules
    normalized = re.sub(r"\.{4,}", " ... ", normalized)
    normalized = re.sub(r"[-_=]{5,}", "---", normalized)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines)
    # Collapse 3+ consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_from_text(text: str, source_format: str = "text") -> ParsedDocument:
    """Parse plain text or Markdown into chapter/heading-aware sections."""
    cleaned = normalize_text(text)
    if not cleaned:
        return ParsedDocument(content="", source_format=source_format, sections=[])

    sections: List[ParsedSection] = []
    current_chapter: Optional[str] = None
    current_lines: List[str] = []

    for line in cleaned.split("\n"):
        stripped = line.strip()
        match = CHAPTER_HEADING_RE.match(stripped) if stripped and len(stripped) <= 100 else None
        if match:
            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append(ParsedSection(content=sec_text, chapter=current_chapter))
                current_lines = []
            current_chapter = (match.group(1) or match.group(2) or stripped).strip()
            current_lines.append(stripped)
        else:
            current_lines.append(line)

    if current_lines:
        sec_text = "\n".join(current_lines).strip()
        if sec_text:
            sections.append(ParsedSection(content=sec_text, chapter=current_chapter))

    if not sections:
        sections = [ParsedSection(content=cleaned, chapter=None)]

    return ParsedDocument(
        content=cleaned,
        source_format=source_format,
        sections=sections,
    )


def extract_from_pdf(file_bytes: bytes) -> ParsedDocument:
    """Extract text, page numbers, and chapter headings from a PDF binary payload."""
    sections: List[ParsedSection] = []
    full_pages: List[str] = []
    title: Optional[str] = None
    author: Optional[str] = None
    current_chapter: Optional[str] = None

    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            meta = doc.metadata or {}
            title = (meta.get("title") or "").strip() or None
            author = (meta.get("author") or "").strip() or None

            for page_idx, page in enumerate(doc):
                raw_page = page.get_text("text") or ""
                cleaned_page = normalize_text(raw_page)
                if not cleaned_page:
                    continue

                # Check first non-empty line for a chapter heading
                first_line = cleaned_page.split("\n", 1)[0].strip()
                match = CHAPTER_HEADING_RE.match(first_line) if len(first_line) <= 100 else None
                if match:
                    current_chapter = (match.group(1) or match.group(2) or first_line).strip()

                sections.append(
                    ParsedSection(
                        content=cleaned_page,
                        chapter=current_chapter or f"Page {page_idx + 1}",
                        page_number=page_idx + 1,
                    )
                )
                full_pages.append(cleaned_page)
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.metadata:
            title = getattr(reader.metadata, "title", None) or None
            author = getattr(reader.metadata, "author", None) or None
        for page_idx, page in enumerate(reader.pages):
            cleaned_page = normalize_text(page.extract_text() or "")
            if not cleaned_page:
                continue
            first_line = cleaned_page.split("\n", 1)[0].strip()
            match = CHAPTER_HEADING_RE.match(first_line) if len(first_line) <= 100 else None
            if match:
                current_chapter = (match.group(1) or match.group(2) or first_line).strip()
            sections.append(
                ParsedSection(
                    content=cleaned_page,
                    chapter=current_chapter or f"Page {page_idx + 1}",
                    page_number=page_idx + 1,
                )
            )
            full_pages.append(cleaned_page)

    full_content = "\n\n".join(full_pages)
    return ParsedDocument(
        content=full_content,
        source_format="pdf",
        title=title,
        author=author,
        sections=sections,
    )


def _extract_html_section(html_bytes: bytes, fallback_chapter: Optional[str] = None) -> Optional[ParsedSection]:
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    heading_tag = soup.find(["h1", "h2", "h3", "title"])
    chapter = heading_tag.get_text(" ", strip=True) if heading_tag else fallback_chapter

    text = normalize_text(soup.get_text("\n"))
    if not text:
        return None
    return ParsedSection(content=text, chapter=chapter)


def extract_from_epub(file_bytes: bytes) -> ParsedDocument:
    """Extract spine-ordered chapters and metadata from an EPUB binary payload."""
    sections: List[ParsedSection] = []
    title: Optional[str] = None
    author: Optional[str] = None

    try:
        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(io.BytesIO(file_bytes))
        title_meta = book.get_metadata("DC", "title")
        if title_meta:
            title = str(title_meta[0][0]).strip() or None
        creator_meta = book.get_metadata("DC", "creator")
        if creator_meta:
            author = str(creator_meta[0][0]).strip() or None

        chapter_idx = 0
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            chapter_idx += 1
            sec = _extract_html_section(item.get_content(), fallback_chapter=f"Section {chapter_idx}")
            if sec:
                sections.append(sec)
    except Exception:
        # Fallback: inspect EPUB as a ZIP container of XHTML/HTML documents
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            html_names = sorted(
                name
                for name in zf.namelist()
                if name.lower().endswith((".xhtml", ".html", ".htm"))
                and "nav" not in name.lower()
                and "toc" not in name.lower()
            )
            for idx, name in enumerate(html_names, start=1):
                sec = _extract_html_section(zf.read(name), fallback_chapter=f"Chapter {idx}")
                if sec:
                    sections.append(sec)

    full_content = "\n\n".join(s.content for s in sections)
    return ParsedDocument(
        content=full_content,
        source_format="epub",
        title=title,
        author=author,
        sections=sections,
    )


def detect_source_format(filename: Optional[str] = None, file_bytes: Optional[bytes] = None) -> str:
    if filename:
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return "pdf"
        if lower.endswith(".epub"):
            return "epub"
        if lower.endswith((".md", ".markdown")):
            return "markdown"
        if lower.endswith(".txt"):
            return "text"

    if file_bytes:
        if file_bytes.startswith(b"%PDF-"):
            return "pdf"
        if file_bytes.startswith(b"PK\x03\x04"):
            return "epub"

    return "text"


def parse_document(
    content: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
) -> ParsedDocument:
    """Unified ingestion parser supporting PDF, EPUB, Markdown, and raw text."""
    fmt = detect_source_format(filename=filename, file_bytes=file_bytes)

    if file_bytes is not None:
        if fmt == "pdf":
            return extract_from_pdf(file_bytes)
        if fmt == "epub":
            return extract_from_epub(file_bytes)
        decoded = file_bytes.decode("utf-8", errors="replace")
        return extract_from_text(decoded, source_format=fmt)

    return extract_from_text(content or "", source_format=fmt)


def build_contextual_embedding_text(
    chunk_text: str,
    title: str,
    author: str,
    chapter: Optional[str] = None,
    page_number: Optional[int] = None,
) -> str:
    """Prepend book, author, and structural metadata header for contextualized embeddings."""
    meta_parts = [f"Book: {title} by {author}"]
    if chapter:
        meta_parts.append(f"Chapter: {chapter}")
    if page_number is not None:
        meta_parts.append(f"Page: {page_number}")
    header = " | ".join(meta_parts)
    return f"{header}\n{chunk_text}"


def chunk_document(
    title: str,
    author: str,
    content: Optional[str] = None,
    parsed_doc: Optional[ParsedDocument] = None,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    token_counter: Callable[[str], int] = estimate_token_count,
) -> List[ChunkPayload]:
    """
    Two-stage hierarchical + token-aware recursive chunker:
      1. Splits along structural section/chapter/page boundaries.
      2. Recursively splits sections to stay within `chunk_size_tokens` (default 200 tokens,
         safely below all-MiniLM-L6-v2's 256-token truncation window) with `chunk_overlap_tokens`.
      3. Injects contextual book/chapter headers into `embedding_text`.
    """
    if parsed_doc is None:
        parsed_doc = extract_from_text(content or "")

    sections = parsed_doc.sections
    if not sections and parsed_doc.content:
        sections = [ParsedSection(content=parsed_doc.content)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=min(chunk_overlap_tokens, max(0, chunk_size_tokens // 2)),
        length_function=token_counter,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
    )

    chunks: List[ChunkPayload] = []
    chunk_index = 0

    for section in sections:
        sec_text = normalize_text(section.content)
        if not sec_text:
            continue

        split_texts = splitter.split_text(sec_text)
        for piece in split_texts:
            cleaned_piece = piece.strip()
            if not cleaned_piece:
                continue
            tokens = token_counter(cleaned_piece)
            embedding_text = build_contextual_embedding_text(
                chunk_text=cleaned_piece,
                title=title,
                author=author,
                chapter=section.chapter,
                page_number=section.page_number,
            )
            chunks.append(
                ChunkPayload(
                    chunk_index=chunk_index,
                    content=cleaned_piece,
                    embedding_text=embedding_text,
                    chapter=section.chapter,
                    page_number=section.page_number,
                    token_count=tokens,
                )
            )
            chunk_index += 1

    return chunks
