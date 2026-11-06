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


@pytest.mark.anyio
async def test_long_book_chunking_and_late_passage_retrieval():
    """Matrix Items 1 & 2: Long books (>256 tokens) are chunked and ranked at passage level."""
    from app.cqrs import BookCommandHandler, CreateBookCommand
    from app.rag import search_similar_books
    from app.models import BookChunk

    chapter_1 = (
        "# Chapter 1: The Harbor Departure\n"
        + ("The merchant vessel sailed across the calm sapphire bay at dawn. " * 35)
    )
    chapter_2 = (
        "# Chapter 2: The Mountain Observatory\n"
        + ("Astronomers calibrated the brass telescope on the frozen summit ridge. " * 35)
    )
    chapter_3 = (
        "# Chapter 3: The Subterranean Vault\n"
        "Deep beneath the obsidian cavern, Cartographer Elena discovered the emerald astrolabe "
        "inscribed with ancient lunar coordinates."
    )
    full_content = f"{chapter_1}\n\n{chapter_2}\n\n{chapter_3}"

    book = await BookCommandHandler.create_book(
        CreateBookCommand(
            title="Chronicles of the Astrolabe",
            author="Elena Vance",
            content=full_content,
            user_id=None,
        )
    )
    assert book.chunk_count >= 3
    chunks = await BookChunk.filter(book_id=book.id).order_by("chunk_index")
    assert len(chunks) == book.chunk_count
    assert all(c.token_count <= 220 for c in chunks)

    # Query specifically about the late passage well past token 256
    results = await search_similar_books(
        "emerald astrolabe obsidian cavern lunar coordinates",
        limit=3,
        book_id=str(book.id),
    )
    assert len(results) >= 1
    assert "emerald astrolabe" in results[0]["content"].lower()
    assert results[0]["chapter"] == "Chapter 3: The Subterranean Vault"


@pytest.mark.anyio
async def test_multi_tenant_isolation_and_sql_injection_guard():
    """Matrix Item 3: user_id scoping prevents cross-user leakage and rejects SQL injection."""
    from app.models import User
    from app.cqrs import BookCommandHandler, CreateBookCommand
    from app.rag import search_similar_books

    user_a = await User.create(username="tenant_a", password_hash="hash_a", role="standard")
    user_b = await User.create(username="tenant_b", password_hash="hash_b", role="standard")

    await BookCommandHandler.create_book(
        CreateBookCommand(
            title="Secret Quantum Ledger",
            author="Alice",
            content="Project Tachyon uses a cryogenic rubidium resonator for quantum key distribution.",
            user_id=str(user_a.id),
        )
    )
    await BookCommandHandler.create_book(
        CreateBookCommand(
            title="Botanical Gardens",
            author="Bob",
            content="Orchids require humid greenhouse conditions and filtered sunlight.",
            user_id=str(user_b.id),
        )
    )

    # User B searching globally must NOT receive User A's private book
    res_b = await search_similar_books(
        "cryogenic rubidium resonator quantum key",
        limit=5,
        user_id=str(user_b.id),
    )
    assert all("rubidium" not in r["content"].lower() for r in res_b)

    # User A searching globally MUST find their own book
    res_a = await search_similar_books(
        "cryogenic rubidium resonator quantum key",
        limit=5,
        user_id=str(user_a.id),
    )
    assert any("rubidium" in r["content"].lower() for r in res_a)

    # SQL injection string in book_id must raise ValueError
    with pytest.raises(ValueError):
        await search_similar_books("orchids", book_id="' OR '1'='1")


@pytest.mark.anyio
async def test_update_book_reindexes_chunks_and_compacts_audit_log():
    """Matrix Item 4: Updating book content refreshes BookChunk vectors and avoids raw payload bloat."""
    from app.cqrs import BookCommandHandler, CreateBookCommand, UpdateBookCommand
    from app.rag import search_similar_books
    from app.models import AuditLog

    book = await BookCommandHandler.create_book(
        CreateBookCommand(
            title="Mutable Tome",
            author="Author X",
            content="Initial text about medieval alchemy and sulfur distillation.",
        )
    )
    updated = await BookCommandHandler.update_book(
        UpdateBookCommand(
            id=str(book.id),
            content="Revised text about superconducting tokamak plasma confinement.",
        )
    )
    assert updated.chunk_count >= 1

    res = await search_similar_books("superconducting tokamak plasma", book_id=str(book.id))
    assert len(res) == 1
    assert "tokamak" in res[0]["content"].lower()
    assert "alchemy" not in res[0]["content"].lower()

    audit = await AuditLog.filter(entity_id=book.id, event_type="BOOK_UPDATED").first()
    assert audit is not None
    assert audit.payload.get("reindexed") is True
    assert "content_sha256" in audit.payload
    assert "content" not in audit.payload


@pytest.mark.anyio
async def test_pdf_and_epub_upload_endpoints():
    """Part 1.1: Multipart file upload supports PDF and EPUB formats and creates chunks."""
    import io
    import zipfile
    import fitz

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post("/users", json={"username": "uploaduser", "password_hash": "uploadpass"})
        login_res = await ac.post("/token", data={"username": "uploaduser", "password": "uploadpass"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Create an in-memory PDF with 2 pages
        pdf_doc = fitz.open()
        p1 = pdf_doc.new_page()
        p1.insert_text((72, 72), "Chapter 1: Solar Sails\nPhotons impart momentum to reflective polyimide membranes.")
        p2 = pdf_doc.new_page()
        p2.insert_text((72, 72), "Chapter 2: Ion Thrusters\nXenon propellant is ionized and accelerated electrostatically.")
        pdf_bytes = pdf_doc.tobytes()
        pdf_doc.close()

        res_pdf = await ac.post(
            "/books/upload",
            files={"file": ("spacecraft_propulsion.pdf", pdf_bytes, "application/pdf")},
            data={"title": "Spacecraft Propulsion", "author": "Dr. Kepler"},
            headers=headers,
        )
        assert res_pdf.status_code == 200
        pdf_book = res_pdf.json()
        assert pdf_book["source_format"] == "pdf"
        assert pdf_book["chunk_count"] >= 2

        chunks_res = await ac.get(f"/books/{pdf_book['id']}/chunks", headers=headers)
        assert chunks_res.status_code == 200
        assert len(chunks_res.json()) >= 2

        # 2. Create an in-memory EPUB container
        epub_buf = io.BytesIO()
        with zipfile.ZipFile(epub_buf, "w") as zf:
            zf.writestr(
                "OEBPS/chapter1.xhtml",
                "<html><head><title>Chapter 1</title></head><body>"
                "<h1>Chapter 1: Coral Reefs</h1>"
                "<p>Symbiotic zooxanthellae provide nutrients to reef-building corals.</p>"
                "</body></html>",
            )
            zf.writestr(
                "OEBPS/chapter2.xhtml",
                "<html><head><title>Chapter 2</title></head><body>"
                "<h2>Chapter 2: Hydrothermal Vents</h2>"
                "<p>Chemosynthetic bacteria thrive near deep ocean black smokers.</p>"
                "</body></html>",
            )
        epub_bytes = epub_buf.getvalue()

        res_epub = await ac.post(
            "/books/upload",
            files={"file": ("marine_biology.epub", epub_bytes, "application/epub+zip")},
            data={"title": "Marine Biology", "author": "Dr. Cousteau"},
            headers=headers,
        )
        assert res_epub.status_code == 200
        epub_book = res_epub.json()
        assert epub_book["source_format"] == "epub"
        assert epub_book["chunk_count"] >= 2


@pytest.mark.anyio
async def test_intent_routing_and_admin_websocket_unlock():
    """Matrix Items 5 & 6: No false escalation on 'humanity'/'help' in book questions; admin WS clears lock."""
    from app.agent import is_escalation_request, agent_app
    from langchain_core.messages import HumanMessage
    from fastapi.testclient import TestClient

    # Legitimate book questions containing 'human' or 'help' substrings must NOT escalate
    assert is_escalation_request("How does the protagonist help humanity in chapter 2?") is False
    assert is_escalation_request("What does the book say about human nature?") is False

    # Explicit escalation requests MUST escalate
    assert is_escalation_request("I need help from a human assistant") is True
    assert is_escalation_request("talk to a human") is True
    assert is_escalation_request("human") is True

    res_normal = await agent_app.ainvoke(
        {
            "messages": [HumanMessage(content="What does the book say about human nature?")],
            "book_id": None,
            "user_id": None,
            "human_required": False,
        }
    )
    assert res_normal["human_required"] is False

    # Now test full escalation + Admin WebSocket unlock flow
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post("/users", json={"username": "esc_user", "password_hash": "pass"})
        await ac.post(
            "/users",
            json={"username": "esc_admin", "password_hash": "pass", "role": "admin"},
            headers={"X-Origin": "admin-ui"},
        )

    user_token = create_access_token(data={"sub": "esc_user"})
    admin_token = create_access_token(data={"sub": "esc_admin"})

    with TestClient(app) as client:
        with client.websocket_connect("/chat") as user_ws:
            user_ws.send_text(user_token)
            assert user_ws.receive_text() == "Authenticated"

            # Trigger escalation
            user_ws.send_text("I need help from a human assistant")
            esc_reply = user_ws.receive_text()
            assert "human assistant" in esc_reply.lower()

            # Find the escalated session
            from app.models import ChatSession
            session = await ChatSession.filter(is_human_required=True).order_by("-created_at").first()
            assert session is not None

            # Admin connects via WebSocket and replies, which must clear is_human_required
            with client.websocket_connect(f"/admin/chat/{session.id}") as admin_ws:
                admin_ws.send_text(admin_token)
                assert admin_ws.receive_text() == "Authenticated"
                admin_ws.send_text("Hello, I have resolved your question!")
                admin_ack = admin_ws.receive_text()
                assert "Admin sent:" in admin_ack

                # User receives admin message
                forwarded = user_ws.receive_text()
                assert "resolved your question" in forwarded

            # Next message from user should now reach the AI agent again instead of being locked
            user_ws.send_text("What is the main theme of the book?")
            unlocked_reply = user_ws.receive_text()
            assert "Human assistant has been notified" not in unlocked_reply


@pytest.mark.anyio
async def test_evaluation_metrics_and_admin_endpoints():
    """Part 1.3: Verify IR & RAGAS metrics and admin evaluation endpoints."""
    from app.evaluation import (
        hit_rate_at_k,
        recall_at_k,
        precision_at_k,
        mrr_at_k,
        ndcg_at_k,
        context_precision_score,
        context_recall_score,
        faithfulness_score,
    )

    retrieved = ["c2", "c1", "c3", "c4", "c5"]
    relevant = {"c1", "c4"}

    assert hit_rate_at_k(retrieved, relevant, k=5) == 1.0
    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    assert precision_at_k(retrieved, relevant, k=5) == 0.4
    assert mrr_at_k(retrieved, relevant, k=5) == 0.5
    assert 0.5 < ndcg_at_k(retrieved, relevant, k=5) < 1.0

    contexts = [
        "Gatsby lived in a colossal mansion in West Egg across the bay from Daisy.",
        "Unrelated passage about gardening tools.",
    ]
    ground_truth = "Gatsby lived in a mansion in West Egg near Daisy."
    answer = "Gatsby lived in a West Egg mansion across the bay from Daisy."

    assert context_precision_score(contexts, ground_truth) == 1.0
    assert context_recall_score(contexts, ground_truth) >= 0.8
    assert faithfulness_score(answer, contexts) >= 0.8

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post(
            "/users",
            json={"username": "eval_admin", "password_hash": "evalpass", "role": "admin"},
            headers={"X-Origin": "admin-ui"},
        )
        login_res = await ac.post("/token", data={"username": "eval_admin", "password": "evalpass"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        summary_res = await ac.get("/admin/evaluation/summary", headers=headers)
        assert summary_res.status_code == 200
        summary = summary_res.json()
        assert "corpus" in summary
        assert "feedback" in summary
        assert "retrieval_telemetry" in summary

        bench_res = await ac.post("/admin/evaluation/benchmark", json={"k": 3, "max_samples": 5}, headers=headers)
        assert bench_res.status_code == 200
        bench = bench_res.json()
        assert "ir_metrics" in bench
        assert "ragas_metrics" in bench
        assert bench["ir_metrics"]["hit_rate_at_k"] >= 0.5


@pytest.mark.anyio
async def test_resume_book_hybrid_retrieval_and_grounded_answers():
    """Verify hybrid retrieval and grounded answer synthesis on multi-chunk resume (Thark, Zanthus, Uninove, Volkswagen)."""
    from app.cqrs import BookCommandHandler, CreateBookCommand
    from app.agent import agent_app, build_contextual_search_query
    from langchain_core.messages import HumanMessage, AIMessage

    resume_content = (
        "Bruno Gomes da Silva\n"
        "Full Stack Technical Leader Engineer on PHP/NodeJs/Python/AI-ML/Angular/React\n"
        "São Paulo-Brazil, Phone: +55 11 98161 6755\n"
        "Education/Certification\n"
        "Uninove, System Information(bachelor) – Dec/2012\n"
        "Thesis: BI applied for Health Care\n"
        "Uninove, Artificial Intelligence (Post-grad) – Dec/2023\n"
        "Uninove, Data Science (Post-grad) – Dec/2024\n"
        "IELTS Academic Training Certified, graded B2 - Nov/2022\n\n"
        + ("Additional technical background and machine learning projects. " * 25)
        + "\n\nExperience\n"
        "Thark, August/2025 – October/2026, contractor\n"
        "Technical Leader Fullstack Engineer Nodejs/Python/React, which 3 pillars:\n"
        "1. technical leadership: I leaded a team of 4 skilled and amazing people and deployed Agile Scrum.\n\n"
        + ("Further project details on microservices and cloud migrations. " * 25)
        + "\n\nZanthus, August/2019 – september/2020 - contractor\n"
        "Sr Full Stack Software Developer, which:\n"
        "1. Through software development I did made a NPS platform connected to a sales automation process.\n\n"
        + ("Trailing experience entries at other companies. " * 15)
        + "\n\nACS(allocated to Apple), January/2014 – August/2016 - Bilingual support."
    )

    book = await BookCommandHandler.create_book(
        CreateBookCommand(
            title="Bruno Silva full resume",
            author="Bruno Silva",
            content=resume_content,
            user_id=None,
        )
    )

    # Standalone short questions with new entities must NOT be polluted by previous questions
    q_history = [
        HumanMessage(content="did the author worked at Volkswagen"),
        AIMessage(content="No."),
        HumanMessage(content="did the author worked at Thark"),
    ]
    assert build_contextual_search_query(q_history) == "did the author worked at Thark"

    # 1. Negative company question (Volkswagen) -> No
    res_vw = await agent_app.ainvoke(
        {
            "messages": [HumanMessage(content="did the author worked at Volkswagen")],
            "book_id": str(book.id),
            "user_id": None,
            "human_required": False,
        }
    )
    ans_vw = res_vw["messages"][-1].content
    assert ans_vw.lower().startswith("no")

    # 2. Positive company question right after a negative answer (Thark) -> Yes + Thark details
    res_thark = await agent_app.ainvoke(
        {
            "messages": [
                HumanMessage(content="did the author worked at Volkswagen"),
                AIMessage(content=ans_vw),
                HumanMessage(content="did the author worked at Thark"),
            ],
            "book_id": str(book.id),
            "user_id": None,
            "human_required": False,
        }
    )
    ans_thark = res_thark["messages"][-1].content
    assert ans_thark.lower().startswith("yes")
    assert "thark" in ans_thark.lower()

    # 3. Positive company question for Zanthus -> Yes + Zanthus details
    res_zanthus = await agent_app.ainvoke(
        {
            "messages": [HumanMessage(content="did the author worked at Zanthus")],
            "book_id": str(book.id),
            "user_id": None,
            "human_required": False,
        }
    )
    ans_zanthus = res_zanthus["messages"][-1].content
    assert ans_zanthus.lower().startswith("yes")
    assert "zanthus" in ans_zanthus.lower()

    # 4. Education question -> Uninove
    res_study = await agent_app.ainvoke(
        {
            "messages": [HumanMessage(content="where Bruno Silva studied")],
            "book_id": str(book.id),
            "user_id": None,
            "human_required": False,
        }
    )
    ans_study = res_study["messages"][-1].content
    assert "uninove" in ans_study.lower()
