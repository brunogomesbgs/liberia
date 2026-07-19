import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.auth import create_access_token
from tortoise import Tortoise
import asyncio

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="module", autouse=True)
async def initialize_tests():
    await Tortoise.init(
        db_url="sqlite://:memory:",
        modules={"models": ["app.models"]}
    )
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()

@pytest.mark.anyio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Liberia API"}

@pytest.mark.anyio
async def test_create_user():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/users", json={"username": "testuser", "password_hash": "testpass"})
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

@pytest.mark.anyio
async def test_login():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create user first
        await ac.post("/users", json={"username": "loginuser", "password_hash": "loginpass"})
        # Login
        response = await ac.post("/token", data={"username": "loginuser", "password": "loginpass"})
    assert response.status_code == 200
    assert "access_token" in response.json()

@pytest.mark.anyio
async def test_cqrs_book_creation():
    from app.cqrs import BookCommandHandler, CreateBookCommand
    from app.models import AuditLog, Book
    
    command = CreateBookCommand(title="Test Book", author="Author", content="Content", user_id=None)
    book = await BookCommandHandler.create_book(command)
    
    assert book.title == "Test Book"
    
    # Check audit log
    audit = await AuditLog.filter(entity_id=book.id).first()
    assert audit is not None
    assert audit.event_type == "BOOK_CREATED"

@pytest.mark.anyio
async def test_books_flow():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create user and login
        await ac.post("/users", json={"username": "bookuser", "password_hash": "bookpass"})
        login_res = await ac.post("/token", data={"username": "bookuser", "password": "bookpass"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create book
        book_data = {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "content": "It was a cool night..."}
        res = await ac.post("/books", json=book_data, headers=headers)
        assert res.status_code == 200
        assert res.json()["title"] == "The Great Gatsby"

@pytest.mark.anyio
async def test_websocket_chat():
    from app.main import app
    from app.auth import create_access_token
    import json

    token = create_access_token(data={"sub": "testuser"})
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create user for authentication in the websocket
        await ac.post("/users", json={"username": "testuser", "password_hash": "testpass"})

    # We need a way to test websocket with AsyncClient
    # Since httpx AsyncClient doesn't support websockets directly easily with ASGITransport for testing 
    # without additional dependencies like 'httpx-ws', we'll simulate the logic or use starlette's TestClient
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        with client.websocket_connect("/chat") as websocket:
            websocket.send_text(token)
            data = websocket.receive_text()
            assert data == "Authenticated"
            
            websocket.send_text("Hello")
            data = websocket.receive_text()
            assert "[Mock]" in data or data # Ensure we get some response
