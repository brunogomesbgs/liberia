import hashlib
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from app.models import Book, BookChunk, AuditLog
from app.ingestion import (
    ParsedDocument,
    parse_document,
    chunk_document,
)
from app.rag import (
    async_embed_documents,
    format_vector_str,
    _is_postgres_connection,
)


# Commands
class CreateBookCommand(BaseModel):
    title: str
    author: str
    content: str
    user_id: Optional[str] = None
    source_format: str = "text"
    filename: Optional[str] = None


class UpdateBookCommand(BaseModel):
    id: str
    title: Optional[str] = None
    author: Optional[str] = None
    content: Optional[str] = None
    user_id: Optional[str] = None


def _build_audit_payload(
    book: Book,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a compact audit payload with SHA-256 content hash and preview rather than
    duplicating full multi-megabyte book text inside `audit_logs.payload`.
    """
    raw_content = book.content or ""
    content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    payload: Dict[str, Any] = {
        "title": book.title,
        "author": book.author,
        "source_format": book.source_format,
        "filename": book.filename,
        "chunk_count": book.chunk_count,
        "content_length": len(raw_content),
        "content_sha256": content_hash,
        "content_preview": raw_content[:200],
        "user_id": str(book.user_id) if book.user_id else None,
    }
    if extra:
        payload.update(extra)
    return payload


class BookCommandHandler:
    @staticmethod
    async def _index_book_chunks(
        book: Book,
        parsed_doc: Optional[ParsedDocument] = None,
    ) -> int:
        chunk_payloads = chunk_document(
            title=book.title,
            author=book.author,
            content=book.content,
            parsed_doc=parsed_doc,
        )
        if not chunk_payloads:
            return 0

        embedding_texts = [c.embedding_text for c in chunk_payloads]
        vectors = await async_embed_documents(embedding_texts)
        vector_strs = [format_vector_str(v) for v in vectors]

        lead_vector_str = vector_strs[0] if vector_strs else None
        conn = Tortoise.get_connection("default")
        is_pg = _is_postgres_connection(conn)
        owner_id = str(book.user_id) if book.user_id else None

        await BookChunk.filter(book_id=book.id).delete()

        for payload, vec_str in zip(chunk_payloads, vector_strs):
            chunk_obj = await BookChunk.create(
                book=book,
                user_id=owner_id,
                chunk_index=payload.chunk_index,
                chapter=payload.chapter,
                page_number=payload.page_number,
                content=payload.content,
                token_count=payload.token_count,
                vector=None if is_pg else vec_str,
            )
            if is_pg:
                await conn.execute_query(
                    "UPDATE book_chunks SET vector = $1::vector WHERE id = $2::uuid",
                    [vec_str, str(chunk_obj.id)],
                )

        book.chunk_count = len(chunk_payloads)
        book.ingestion_status = "completed"
        if not is_pg:
            book.vector = lead_vector_str
        await book.save()

        if is_pg and lead_vector_str:
            await conn.execute_query(
                "UPDATE books SET vector = $1::vector WHERE id = $2::uuid",
                [lead_vector_str, str(book.id)],
            )

        return len(chunk_payloads)

    @staticmethod
    async def create_book(
        command: CreateBookCommand,
        parsed_doc: Optional[ParsedDocument] = None,
    ) -> Book:
        if parsed_doc is None:
            parsed_doc = parse_document(
                content=command.content,
                filename=command.filename,
            )

        normalized_content = parsed_doc.content if parsed_doc.content else command.content
        source_format = parsed_doc.source_format or command.source_format or "text"

        async with in_transaction():
            book = await Book.create(
                title=command.title,
                author=command.author,
                content=normalized_content,
                user_id=command.user_id,
                source_format=source_format,
                filename=command.filename,
                ingestion_status="processing",
            )
            await BookCommandHandler._index_book_chunks(book, parsed_doc=parsed_doc)

            await AuditLog.create(
                event_type="BOOK_CREATED",
                entity_id=book.id,
                payload=_build_audit_payload(book),
                user_id=command.user_id,
            )
        return book

    @staticmethod
    async def ingest_uploaded_book(
        file_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Book:
        parsed_doc = parse_document(file_bytes=file_bytes, filename=filename)
        stem = filename.rsplit(".", 1)[0] if filename else "Untitled Book"
        resolved_title = (title or parsed_doc.title or stem).strip() or "Untitled Book"
        resolved_author = (author or parsed_doc.author or "Unknown Author").strip() or "Unknown Author"

        command = CreateBookCommand(
            title=resolved_title,
            author=resolved_author,
            content=parsed_doc.content,
            user_id=user_id,
            source_format=parsed_doc.source_format,
            filename=filename,
        )
        return await BookCommandHandler.create_book(command, parsed_doc=parsed_doc)

    @staticmethod
    async def update_book(command: UpdateBookCommand) -> Book:
        async with in_transaction():
            book = await Book.get(id=command.id)
            update_data = (
                command.model_dump(exclude_unset=True, exclude={"id", "user_id"})
                if hasattr(command, "model_dump")
                else command.dict(exclude_unset=True, exclude={"id", "user_id"})
            )
            needs_reindex = any(
                field in update_data and getattr(book, field) != update_data[field]
                for field in ("title", "author", "content")
            )

            book.update_from_dict(update_data)
            await book.save()

            if needs_reindex:
                parsed_doc = parse_document(content=book.content, filename=book.filename)
                book.content = parsed_doc.content or book.content
                await BookCommandHandler._index_book_chunks(book, parsed_doc=parsed_doc)

            await AuditLog.create(
                event_type="BOOK_UPDATED",
                entity_id=book.id,
                payload=_build_audit_payload(
                    book,
                    extra={
                        "updated_fields": sorted(update_data.keys()),
                        "reindexed": needs_reindex,
                    },
                ),
                user_id=command.user_id,
            )
        return book


# Queries
class BookQueryHandler:
    @staticmethod
    async def get_book(book_id: str, user_id: Optional[str] = None):
        if user_id:
            return await Book.get_or_none(id=book_id, user_id=user_id)
        return await Book.get_or_none(id=book_id)

    @staticmethod
    async def list_books(user_id: Optional[str] = None):
        if user_id:
            return await Book.filter(user_id=user_id).order_by("-created_at")
        return await Book.all().order_by("-created_at")

    @staticmethod
    async def list_book_chunks(book_id: str):
        return await BookChunk.filter(book_id=book_id).order_by("chunk_index")
