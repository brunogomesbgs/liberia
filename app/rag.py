import os
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from tortoise import Tortoise
from .models import Book

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize free embeddings
# Using a mock or conditional loading for tests to avoid heavy downloads
try:
    if os.getenv("TESTING") == "1":
        class MockEmbeddings:
            def embed_query(self, text): return [0.0] * 384
            def embed_documents(self, texts): return [[0.0] * 384] * len(texts)
        embeddings = MockEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL_ID", "all-MiniLM-L6-v2"))
except Exception:
    class MockEmbeddings:
        def embed_query(self, text): return [0.0] * 384
        def embed_documents(self, texts): return [[0.0] * 384] * len(texts)
    embeddings = MockEmbeddings()
# all-MiniLM-L6-v2 has 384 dimensions

async def init_vector_extension():
    # Get connection
    conn = Tortoise.get_connection("default")
    # Enable pgvector extension
    await conn.execute_script("CREATE EXTENSION IF NOT EXISTS vector;")
    # Add vector column to books table if it doesn't exist (Self-healing part)
    # all-MiniLM-L6-v2 dimensions = 384
    try:
        await conn.execute_script("ALTER TABLE books ALTER COLUMN vector TYPE vector(384) USING vector::vector(384);")
    except Exception as e:
        print(f"Vector column alteration: {e}")

async def add_book_with_vector(title: str, author: str, content: str, vector: list):
    # This would be part of the CQRS handler ideally
    conn = Tortoise.get_connection("default")
    vector_str = str(vector).replace(' ', '')
    sql = "INSERT INTO books (id, title, author, content, vector, created_at, updated_at) VALUES (uuid_generate_v7(), $1, $2, $3, $4, now(), now())"
    # Actually we use uuid6 in python, let's stick to tortoise create if possible or raw sql
    pass

async def search_similar_books(query: str, limit: int = 5, book_id: str = None):
    query_vector = embeddings.embed_query(query)
    conn = Tortoise.get_connection("default")
    vector_str = str(query_vector).replace(' ', '')
    
    where_clause = ""
    if book_id:
        where_clause = f"WHERE id = '{book_id}'"
        
    sql = f"SELECT id, title, author, content FROM books {where_clause} ORDER BY vector <=> '{vector_str}' LIMIT {limit}"
    return await conn.execute_query_dict(sql)
