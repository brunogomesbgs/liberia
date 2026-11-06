from dataclasses import dataclass, field
import math
import re
from typing import List, Set, Sequence, Dict, Any, Optional

from app.models import Book, BookChunk, ChatMessage, ChatSession
from app.rag import search_similar_chunks, async_embed_query, cosine_distance


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "it", "this", "that", "these", "those", "what", "who",
    "where", "when", "why", "how", "did", "does", "do",
}


def _content_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return filtered if filtered else words


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
    """1.0 if at least one relevant document/chunk appears in top-k results, else 0.0."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Proportion of all relevant chunks retrieved within top-k."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(list(retrieved_ids)[:k])
    hits = len(top_k.intersection(relevant_ids))
    return float(hits) / float(len(relevant_ids))


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return float(hits) / float(len(top_k))


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Reciprocal rank (1 / rank) of the first relevant chunk in top-k."""
    if not relevant_ids or k <= 0:
        return 0.0
    for idx, doc_id in enumerate(list(retrieved_ids)[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / float(idx)
    return 0.0


def average_precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Average Precision at k (AP@k), used to compute MAP@k across queries."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    hits = 0
    sum_precisions = 0.0
    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precisions += float(hits) / float(idx)
    if hits == 0:
        return 0.0
    denom = min(len(relevant_ids), k)
    return sum_precisions / float(denom)


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k (nDCG@k) with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1.0)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(idx + 1.0) for idx in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def context_precision_score(
    retrieved_contexts: Sequence[str],
    ground_truth: str,
    retrieved_ids: Optional[Sequence[str]] = None,
    relevant_ids: Optional[Set[str]] = None,
) -> float:
    """
    RAGAS-aligned Context Precision: measures signal-to-noise ordering of retrieved contexts.
    Uses ground-truth chunk IDs when available, otherwise evaluates token overlap per rank.
    """
    if retrieved_ids is not None and relevant_ids:
        return average_precision_at_k(retrieved_ids, relevant_ids, k=max(1, len(retrieved_ids)))

    if not retrieved_contexts:
        return 0.0
    gt_tokens = set(_content_tokens(ground_truth))
    if not gt_tokens:
        return 0.0

    relevance_flags: List[int] = []
    for ctx in retrieved_contexts:
        ctx_tokens = set(_content_tokens(ctx))
        overlap = len(gt_tokens.intersection(ctx_tokens)) / float(len(gt_tokens))
        relevance_flags.append(1 if overlap >= 0.25 else 0)

    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0

    hits = 0
    weighted_sum = 0.0
    for idx, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            hits += 1
            weighted_sum += float(hits) / float(idx)
    return weighted_sum / float(total_relevant)


def context_recall_score(retrieved_contexts: Sequence[str], ground_truth: str) -> float:
    """
    RAGAS-aligned Context Recall: fraction of ground-truth claims/key terms supported by retrieved context.
    """
    gt_tokens = set(_content_tokens(ground_truth))
    if not gt_tokens:
        return 1.0
    combined_context = " ".join(retrieved_contexts or [])
    ctx_tokens = set(_content_tokens(combined_context))
    if not ctx_tokens:
        return 0.0
    covered = len(gt_tokens.intersection(ctx_tokens))
    return float(covered) / float(len(gt_tokens))


def faithfulness_score(answer: str, retrieved_contexts: Sequence[str]) -> float:
    """
    RAGAS-aligned Faithfulness: proportion of substantive claims/tokens in the generated answer
    that are grounded in the retrieved context.
    """
    ans_tokens = set(_content_tokens(answer))
    if not ans_tokens:
        return 0.0
    combined_context = " ".join(retrieved_contexts or [])
    ctx_tokens = set(_content_tokens(combined_context))
    if not ctx_tokens:
        return 0.0
    grounded = len(ans_tokens.intersection(ctx_tokens))
    return float(grounded) / float(len(ans_tokens))


async def answer_relevancy_score(question: str, answer: str) -> float:
    """
    RAGAS-aligned Answer Relevancy: semantic embedding alignment + lexical coverage between question and answer.
    """
    if not question or not answer:
        return 0.0
    q_vec = await async_embed_query(question)
    a_vec = await async_embed_query(answer)
    emb_sim = max(0.0, 1.0 - cosine_distance(q_vec, a_vec))

    q_tokens = set(_content_tokens(question))
    a_tokens = set(_content_tokens(answer))
    lex_overlap = (
        len(q_tokens.intersection(a_tokens)) / float(len(q_tokens))
        if q_tokens
        else 0.0
    )
    return min(1.0, round(0.6 * emb_sim + 0.4 * lex_overlap, 4))


@dataclass
class EvaluationSample:
    question: str
    ground_truth: str
    relevant_chunk_ids: Set[str] = field(default_factory=set)
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_contexts: List[str] = field(default_factory=list)
    answer: str = ""
    book_id: Optional[str] = None


async def evaluate_rag_samples(
    samples: Sequence[EvaluationSample],
    k: int = 5,
    use_ragas_library: bool = False,
) -> Dict[str, Any]:
    """
    Compute full IR + RAGAS evaluation metrics across a list of `EvaluationSample` items.
    If `use_ragas_library=True` and the `ragas` + `datasets` libraries are installed with an active LLM,
    augments the report with native `ragas.evaluate` scores.
    """
    if not samples:
        return {
            "sample_count": 0,
            "k": k,
            "ir_metrics": {
                "hit_rate_at_k": 0.0,
                "recall_at_k": 0.0,
                "precision_at_k": 0.0,
                "mrr_at_k": 0.0,
                "map_at_k": 0.0,
                "ndcg_at_k": 0.0,
            },
            "ragas_metrics": {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
            },
            "per_sample": [],
        }

    hit_rates: List[float] = []
    recalls: List[float] = []
    precisions: List[float] = []
    mrrs: List[float] = []
    aps: List[float] = []
    ndcgs: List[float] = []

    ctx_precisions: List[float] = []
    ctx_recalls: List[float] = []
    faithfulness_list: List[float] = []
    ans_relevancies: List[float] = []
    per_sample_reports: List[Dict[str, Any]] = []

    for s in samples:
        hr = hit_rate_at_k(s.retrieved_chunk_ids, s.relevant_chunk_ids, k=k)
        rec = recall_at_k(s.retrieved_chunk_ids, s.relevant_chunk_ids, k=k)
        prec = precision_at_k(s.retrieved_chunk_ids, s.relevant_chunk_ids, k=k)
        mrr = mrr_at_k(s.retrieved_chunk_ids, s.relevant_chunk_ids, k=k)
        ap = average_precision_at_k(s.retrieved_chunk_ids, s.relevant_chunk_ids, k=k)
        ndcg = ndcg_at_k(s.retrieved_chunk_ids, s.relevant_chunk_ids, k=k)

        cp = context_precision_score(
            s.retrieved_contexts,
            s.ground_truth,
            retrieved_ids=s.retrieved_chunk_ids[:k],
            relevant_ids=s.relevant_chunk_ids,
        )
        cr = context_recall_score(s.retrieved_contexts, s.ground_truth)
        answer_text = s.answer or s.ground_truth
        faith = faithfulness_score(answer_text, s.retrieved_contexts)
        ans_rel = await answer_relevancy_score(s.question, answer_text)

        hit_rates.append(hr)
        recalls.append(rec)
        precisions.append(prec)
        mrrs.append(mrr)
        aps.append(ap)
        ndcgs.append(ndcg)

        ctx_precisions.append(cp)
        ctx_recalls.append(cr)
        faithfulness_list.append(faith)
        ans_relevancies.append(ans_rel)

        per_sample_reports.append(
            {
                "question": s.question,
                "book_id": s.book_id,
                "hit_rate_at_k": round(hr, 4),
                "recall_at_k": round(rec, 4),
                "precision_at_k": round(prec, 4),
                "mrr_at_k": round(mrr, 4),
                "ndcg_at_k": round(ndcg, 4),
                "context_precision": round(cp, 4),
                "context_recall": round(cr, 4),
                "faithfulness": round(faith, 4),
                "answer_relevancy": round(ans_rel, 4),
            }
        )

    n = float(len(samples))
    ragas_summary = {
        "context_precision": round(sum(ctx_precisions) / n, 4),
        "context_recall": round(sum(ctx_recalls) / n, 4),
        "faithfulness": round(sum(faithfulness_list) / n, 4),
        "answer_relevancy": round(sum(ans_relevancies) / n, 4),
    }

    if use_ragas_library:
        try:
            from datasets import Dataset  # type: ignore
            from ragas import evaluate as ragas_evaluate  # type: ignore
            from ragas.metrics import (  # type: ignore
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            )

            ds = Dataset.from_dict(
                {
                    "question": [s.question for s in samples],
                    "contexts": [s.retrieved_contexts for s in samples],
                    "answer": [s.answer or s.ground_truth for s in samples],
                    "ground_truth": [s.ground_truth for s in samples],
                }
            )
            ragas_res = ragas_evaluate(
                ds,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            )
            for key in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
                if key in ragas_res and ragas_res[key] is not None:
                    ragas_summary[key] = round(float(ragas_res[key]), 4)
        except Exception:
            pass

    return {
        "sample_count": len(samples),
        "k": k,
        "ir_metrics": {
            "hit_rate_at_k": round(sum(hit_rates) / n, 4),
            "recall_at_k": round(sum(recalls) / n, 4),
            "precision_at_k": round(sum(precisions) / n, 4),
            "mrr_at_k": round(sum(mrrs) / n, 4),
            "map_at_k": round(sum(aps) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
        },
        "ragas_metrics": ragas_summary,
        "per_sample": per_sample_reports,
    }


async def run_retrieval_benchmark(
    k: int = 5,
    book_id: Optional[str] = None,
    max_samples: int = 10,
) -> Dict[str, Any]:
    """
    Run an automated retrieval & RAG evaluation benchmark over the indexed `BookChunk` corpus.
    Synthesizes targeted passage questions from indexed book chunks and evaluates top-k retrieval.
    """
    qs = BookChunk.all().prefetch_related("book").order_by("chunk_index")
    if book_id:
        qs = qs.filter(book_id=book_id)
    chunks = await qs.limit(max_samples)

    samples: List[EvaluationSample] = []
    for ch in chunks:
        words = _content_tokens(ch.content)
        if not words:
            continue
        key_terms = " ".join(words[:10])
        book_title = ch.book.title if ch.book else "the book"
        question = f"In {book_title}, what is discussed regarding {key_terms}?"

        retrieved = await search_similar_chunks(
            query=question,
            limit=k,
            book_id=str(ch.book_id) if book_id else None,
            user_id=ch.user_id,
        )
        retrieved_ids = [str(r["id"]) for r in retrieved]
        retrieved_contexts = [str(r["content"]) for r in retrieved]
        top_answer = retrieved_contexts[0] if retrieved_contexts else ""

        samples.append(
            EvaluationSample(
                question=question,
                ground_truth=ch.content,
                relevant_chunk_ids={str(ch.id)},
                retrieved_chunk_ids=retrieved_ids,
                retrieved_contexts=retrieved_contexts,
                answer=top_answer,
                book_id=str(ch.book_id),
            )
        )

    return await evaluate_rag_samples(samples, k=k)


async def get_online_evaluation_summary() -> Dict[str, Any]:
    """
    Aggregate online human-in-the-loop (HITL) and retrieval provenance metrics across sessions and messages.
    """
    total_Books = await Book.all().count()
    total_chunks = await BookChunk.all().count()
    total_sessions = await ChatSession.all().count()
    escalated_sessions = await ChatSession.filter(is_human_required=True).count()

    all_sessions = await ChatSession.all()
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for s in all_sessions:
        if s.sentiment in sentiment_counts:
            sentiment_counts[s.sentiment] += 1

    agent_messages = await ChatMessage.filter(sender_type__in=["agent", "admin"]).order_by("-created_at")
    up_count = 0
    down_count = 0
    sim_scores: List[float] = []
    up_sim_scores: List[float] = []
    down_sim_scores: List[float] = []
    low_performing_queries: List[Dict[str, Any]] = []

    for m in agent_messages:
        if m.top_similarity_score is not None:
            sim_scores.append(float(m.top_similarity_score))
        if m.feedback == "up":
            up_count += 1
            if m.top_similarity_score is not None:
                up_sim_scores.append(float(m.top_similarity_score))
        elif m.feedback == "down":
            down_count += 1
            if m.top_similarity_score is not None:
                down_sim_scores.append(float(m.top_similarity_score))
            if len(low_performing_queries) < 10:
                low_performing_queries.append(
                    {
                        "message_id": str(m.id),
                        "session_id": str(m.session_id),
                        "content_preview": (m.content or "")[:120],
                        "retrieved_chunk_ids": m.retrieved_chunk_ids or [],
                        "top_similarity_score": m.top_similarity_score,
                    }
                )

    evaluated_count = up_count + down_count
    approval_rate = round(float(up_count) / float(evaluated_count), 4) if evaluated_count > 0 else None
    avg_sim = round(sum(sim_scores) / float(len(sim_scores)), 4) if sim_scores else None
    avg_up_sim = round(sum(up_sim_scores) / float(len(up_sim_scores)), 4) if up_sim_scores else None
    avg_down_sim = round(sum(down_sim_scores) / float(len(down_sim_scores)), 4) if down_sim_scores else None

    return {
        "corpus": {
            "total_books": total_Books,
            "total_chunks": total_chunks,
        },
        "feedback": {
            "evaluated_messages": evaluated_count,
            "thumbs_up": up_count,
            "thumbs_down": down_count,
            "approval_rate": approval_rate,
        },
        "retrieval_telemetry": {
            "avg_top_similarity": avg_sim,
            "avg_thumbs_up_similarity": avg_up_sim,
            "avg_thumbs_down_similarity": avg_down_sim,
        },
        "sessions": {
            "total_sessions": total_sessions,
            "active_escalations": escalated_sessions,
            "sentiment_distribution": sentiment_counts,
        },
        "low_performing_responses": low_performing_queries,
    }
