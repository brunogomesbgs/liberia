from typing import List, Optional
from pydantic import BaseModel
from app.models import Book, AuditLog
import json

# Commands
class CreateBookCommand(BaseModel):
    title: str
    author: str
    content: str
    user_id: Optional[str] = None

class UpdateBookCommand(BaseModel):
    id: str
    title: Optional[str] = None
    author: Optional[str] = None
    content: Optional[str] = None
    user_id: Optional[str] = None

from app.rag import embeddings
from tortoise import Tortoise

# Handlers
class BookCommandHandler:
    @staticmethod
    async def create_book(command: CreateBookCommand):
        # Generate vector for RAG
        vector = embeddings.embed_query(command.content)
        vector_str = str(vector).replace(' ', '')
        
        # Create book using Tortoise
        book = await Book.create(**command.dict())
        
        # Update vector column using raw SQL since Tortoise doesn't support pgvector directly easily
        conn = Tortoise.get_connection("default")
        # Ensure we use string representation of UUID for SQL
        await conn.execute_script(f"UPDATE books SET vector = '{vector_str}' WHERE id = '{str(book.id)}'")

        # Audit log (Event Sourcing - Write flow)
        await AuditLog.create(
            event_type="BOOK_CREATED",
            entity_id=book.id,
            payload=command.dict(),
            user_id=command.user_id
        )
        return book

    @staticmethod
    async def update_book(command: UpdateBookCommand):
        book = await Book.get(id=command.id)
        update_data = command.dict(exclude_unset=True, exclude={"id", "user_id"})
        book.update_from_dict(update_data)
        await book.save()
        
        # Audit log (Event Sourcing - Update flow)
        await AuditLog.create(
            event_type="BOOK_UPDATED",
            entity_id=book.id,
            payload=command.dict(exclude_unset=True),
            user_id=command.user_id
        )
        return book

# Queries
class BookQueryHandler:
    @staticmethod
    async def get_book(book_id: str):
        return await Book.get_or_none(id=book_id)

    @staticmethod
    async def list_books(user_id: Optional[str] = None):
        if user_id:
            return await Book.filter(user_id=user_id)
        return await Book.all()
