import math
import json
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from tortoise import Tortoise

from app.main import app
from app.auth import create_access_token
from app.cqrs import BookCommandHandler, CreateBookCommand
from app.models import Book, BookChunk, ChatSession, ChatMessage, User
from app.evaluation import (
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    average_precision_at_k,
    ndcg_at_k,
    context_precision_score,
    context_recall_score,
    faithfulness_score,
    answer_relevancy_score,
    EvaluationSample,
    evaluate_rag_samples,
    run_retrieval_benchmark,
    get_online_evaluation_summary,
)


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
async def initialize_evaluation_tests():
    await Tortoise.init(
        db_url="sqlite://:memory:",
        modules={"models": ["app.models"]},
    )
    await Tortoise.generate_schemas()
    yield
    await Tortoise.close_connections()


@pytest.mark.anyio
async def test_ir_metrics_accuracy_and_edge_cases():
    retrieved = ["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"]
    relevant = {"chunk_b", "chunk_d"}

    # Standard top-5 evaluation
    assert hit_rate_at_k(retrieved, relevant, k=5) == 1.0
    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2.0 / 5.0)
    assert mrr_at_k(retrieved, relevant, k=5) == pytest.approx(1.0 / 2.0)

    # AP@5: hits at rank 2 (1/2) and rank 4 (2/4), divided by min(2, 5) = 2 -> (0.5 + 0.5) / 2 = 0.5
    assert average_precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)

    # nDCG@5: DCG = 1/log2(3) + 1/log2(5), IDCG = 1/log2(2) + 1/log2(3)
    expected_dcg = (1.0 / math.log2(3.0)) + (1.0 / math.log2(5.0))
    expected_idcg = (1.0 / math.log2(2.0)) + (1.0 / math.log2(3.0))
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(expected_dcg / expected_idcg)

    # Top-1 cutoff (chunk_a is not relevant)
    assert hit_rate_at_k(retrieved, relevant, k=1) == 0.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0
    assert precision_at_k(retrieved, relevant, k=1) == 0.0
    assert mrr_at_k(retrieved, relevant, k=1) == 0.0
    assert average_precision_at_k(retrieved, relevant, k=1) == 0.0
    assert ndcg_at_k(retrieved, relevant, k=1) == 0.0

    # Perfect ranking at top-2
    perfect_retrieved = ["chunk_b", "chunk_d", "chunk_a"]
    assert mrr_at_k(perfect_retrieved, relevant, k=3) == 1.0
    assert average_precision_at_k(perfect_retrieved, relevant, k=3) == 1.0
    assert ndcg_at_k(perfect_retrieved, relevant, k=3) == pytest.approx(1.0)

    # Edge cases: empty inputs or non-positive k
    assert hit_rate_at_k([], relevant, k=5) == 0.0
    assert hit_rate_at_k(retrieved, set(), k=5) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=0) == 0.0
    assert precision_at_k([], relevant, k=5) == 0.0
    assert recall_at_k(retrieved, set(), k=5) == 0.0


@pytest.mark.anyio
async def test_ragas_metrics_precision_recall_faithfulness_relevancy():
    ground_truth = "Photosynthesis converts light energy into chemical energy inside chloroplasts."
    relevant_ctx = "Inside plant chloroplasts, photosynthesis converts light energy into chemical glucose energy."
    noise_ctx = "Medieval castles featured fortified stone curtain walls and deep moats."

    # Context Precision rewards ranking the relevant context at rank 1 over rank 2
    cp_rank1 = context_precision_score([relevant_ctx, noise_ctx], ground_truth)
    cp_rank2 = context_precision_score([noise_ctx, relevant_ctx], ground_truth)
    assert cp_rank1 == pytest.approx(1.0)
    assert cp_rank2 == pytest.approx(0.5)
    assert cp_rank1 > cp_rank2
    assert context_precision_score([], ground_truth) == 0.0

    # Context Recall measures coverage of ground truth tokens
    cr_full = context_recall_score([relevant_ctx], ground_truth)
    cr_none = context_recall_score([noise_ctx], ground_truth)
    assert cr_full >= 0.85
    assert cr_none == 0.0
    assert context_recall_score([], "") == 1.0

    # Faithfulness detects grounded vs hallucinated claims
    faithful_answer = "Photosynthesis inside chloroplasts converts light energy into chemical energy."
    hallucinated_answer = "Supersonic jet engines burn kerosene fuel at high altitude."
    assert faithfulness_score(faithful_answer, [relevant_ctx]) >= 0.85
    assert faithfulness_score(hallucinated_answer, [relevant_ctx]) == 0.0
    assert faithfulness_score("", [relevant_ctx]) == 0.0

    # Answer Relevancy combines embedding similarity and lexical overlap
    question = "How does photosynthesis convert light energy in chloroplasts?"
    rel_high = await answer_relevancy_score(question, faithful_answer)
    rel_low = await answer_relevancy_score(question, hallucinated_answer)
    assert rel_high > rel_low
    assert await answer_relevancy_score("", faithful_answer) == 0.0


@pytest.mark.anyio
async def test_evaluate_rag_samples_aggregation():
    empty_report = await evaluate_rag_samples([], k=3)
    assert empty_report["sample_count"] == 0
    assert empty_report["ir_metrics"]["hit_rate_at_k"] == 0.0
    assert empty_report["ragas_metrics"]["faithfulness"] == 0.0

    samples = [
        EvaluationSample(
            question="What element is used in the atomic clock?",
            ground_truth="Cesium-133 atoms regulate the atomic clock frequency.",
            relevant_chunk_ids={"c1"},
            retrieved_chunk_ids=["c1", "c2"],
            retrieved_contexts=[
                "Cesium-133 atoms regulate the atomic clock frequency.",
                " unrelated text ",
            ],
            answer="Cesium-133 atoms regulate the atomic clock frequency.",
            book_id="book-1",
        ),
        EvaluationSample(
            question="What powers the geothermal plant?",
            ground_truth="Subterranean hydrothermal steam drives the geothermal turbines.",
            relevant_chunk_ids={"c9"},
            retrieved_chunk_ids=["c8", "c9"],
            retrieved_contexts=[
                "Unrelated intro paragraph.",
                "Subterranean hydrothermal steam drives the geothermal turbines.",
            ],
            answer="Subterranean hydrothermal steam drives the geothermal turbines.",
            book_id="book-2",
        ),
    ]

    report = await evaluate_rag_samples(samples, k=2)
    assert report["sample_count"] == 2
    assert report["ir_metrics"]["hit_rate_at_k"] == pytest.approx(1.0)
    assert report["ir_metrics"]["recall_at_k"] == pytest.approx(1.0)
    assert report["ir_metrics"]["precision_at_k"] == pytest.approx(0.5)
    # Sample 1 MRR = 1.0, Sample 2 MRR = 0.5 -> Mean = 0.75
    assert report["ir_metrics"]["mrr_at_k"] == pytest.approx(0.75)
    assert report["ragas_metrics"]["context_recall"] >= 0.9
    assert report["ragas_metrics"]["faithfulness"] >= 0.9
    assert len(report["per_sample"]) == 2


@pytest.mark.anyio
async def test_run_retrieval_benchmark_and_online_telemetry_summary():
    user = await User.create(username="eval_bench_user", password_hash="hash", role="standard")
    admin = await User.create(username="eval_bench_admin", password_hash="hash", role="admin")

    book = await BookCommandHandler.create_book(
        CreateBookCommand(
            title="Encyclopedia of Metallurgy",
            author="Dr. Bessemer",
            content=(
                "# Chapter 1: Smelting Iron\n"
                "Blast furnaces reduce iron ore using coke and limestone flux at high temperatures.\n\n"
                "# Chapter 2: Austenitic Steel\n"
                "Chromium and nickel additions stabilize the face-centered cubic austenite crystal structure."
            ),
            user_id=str(user.id),
        )
    )

    benchmark = await run_retrieval_benchmark(k=3, book_id=str(book.id), max_samples=5)
    assert benchmark["sample_count"] >= 2
    assert benchmark["ir_metrics"]["hit_rate_at_k"] >= 0.5
    assert benchmark["ragas_metrics"]["context_recall"] >= 0.5

    # Simulate chat session + feedback via WebSocket to verify online telemetry aggregation
    user_token = create_access_token(data={"sub": user.username})
    with TestClient(app) as client:
        with client.websocket_connect(f"/chat?bookId={book.id}") as ws:
            ws.send_text(user_token)
            assert ws.receive_text() == "Authenticated"

            ws.send_text("How do blast furnaces reduce iron ore?")
            agent_reply_1 = ws.receive_text()
            assert agent_reply_1

            # Send thumbs-up on the agent message (index 1 in session: 0=user, 1=agent)
            ws.send_text(
                json.dumps(
                    {
                        "type": "feedback",
                        "messageIndex": 1,
                        "totalMessages": 2,
                        "feedback": "up",
                    }
                )
            )

            ws.send_text("What stabilizes austenite crystal structure?")
            agent_reply_2 = ws.receive_text()
            assert agent_reply_2

            # Send thumbs-down on the second agent message (index 3 in session)
            ws.send_text(
                json.dumps(
                    {
                        "type": "feedback",
                        "messageIndex": 3,
                        "totalMessages": 4,
                        "feedback": "down",
                    }
                )
            )

            # Send one more turn so the FIFO WebSocket loop processes the preceding feedback before disconnect
            ws.send_text("Summarize both chapters.")
            agent_reply_3 = ws.receive_text()
            assert agent_reply_3

    summary = await get_online_evaluation_summary()
    assert summary["corpus"]["total_books"] >= 1
    assert summary["corpus"]["total_chunks"] >= 2
    assert summary["feedback"]["evaluated_messages"] >= 2
    assert summary["feedback"]["thumbs_up"] >= 1
    assert summary["feedback"]["thumbs_down"] >= 1
    assert summary["feedback"]["approval_rate"] is not None
    assert summary["retrieval_telemetry"]["avg_top_similarity"] is not None
    assert len(summary["low_performing_responses"]) >= 1

    # Verify RBAC on admin evaluation endpoints (standard user -> 403, admin -> 200)
    admin_token = create_access_token(data={"sub": admin.username})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        forbidden_res = await ac.get(
            "/admin/evaluation/summary",
            headers={"Authorization": f"Bearer {user_token}"},
        )
        assert forbidden_res.status_code == 403

        ok_summary = await ac.get(
            "/admin/evaluation/summary",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert ok_summary.status_code == 200

        ok_bench = await ac.post(
            "/admin/evaluation/benchmark",
            json={"k": 2, "book_id": str(book.id), "max_samples": 3},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert ok_bench.status_code == 200
        assert ok_bench.json()["sample_count"] >= 2
