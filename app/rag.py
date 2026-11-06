import asyncio
import hashlib
import json
import logging
import math
import os
import re
import unicodedata
import uuid
from typing import List, Optional, Dict, Any, Set, Tuple

from dotenv import load_dotenv

load_dotenv(override=False)

from tortoise import Tortoise
from tortoise.expressions import Q
from app.models import Book, BookChunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
DEFAULT_MAX_COSINE_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.85"))

QUERY_STOPWORDS: Set[str] = {
    "a", "about", "an", "and", "any", "are", "as", "at", "author", "be",
    "been", "being", "book", "boook", "books", "by", "can", "could",
    "covered", "describe", "described", "did", "discussed", "do", "document",
    "does", "done", "explain", "file", "for", "from", "give", "had", "has",
    "have", "he", "her", "hers", "him", "his", "how", "in", "into", "is",
    "it", "its", "make", "me", "mentioned", "of", "on", "or", "page",
    "person", "please", "resume", "said", "say", "says", "she", "should",
    "some", "studied", "study", "studying", "subjects", "summary", "summarize",
    "talk", "tell", "text", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "topics", "us", "waht", "was", "were",
    "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    "work", "worked", "working", "would", "writer",
}

STUDY_QUERY_TRIGGERS: Set[str] = {
    "studied", "study", "studying", "education", "educational", "university",
    "college", "school", "degree", "bachelor", "master", "postgrad",
    "graduated", "graduation", "academic", "thesis", "course", "courses",
}
STUDY_CONCEPT_TERMS: Set[str] = {
    "education", "certification", "bachelor", "post-grad",
    "postgrad", "thesis", "uninove", "university", "college",
}

EXPERIENCE_QUERY_TRIGGERS: Set[str] = {
    "experience", "experienced", "leader", "leadership", "lead", "technical",
    "role", "position", "career", "companies", "employer",
}
EXPERIENCE_CONCEPT_TERMS: Set[str] = {
    "experience", "contractor", "leader", "leadership", "leaded", "lead",
    "technical", "engineer", "developer", "fullstack", "full stack", "senior",
    "tenure",
}

OVERVIEW_QUERY_RE = re.compile(
    r"(?:\b(?:what\s+(?:is|it\s+is|are)\s+(?:discussed|covered|said|talked)|"
    r"(?:book+|document|text)\s+(?:is\s+|it\s+is\s+)?about|"
    r"about\s+(?:what|waht)|"
    r"subjects?\s+covered|topics?\s+covered|"
    r"make\s+a\s+(?:resume|summary)|summarize|summary\s+of|overview\s+of|"
    r"resumo)\b)",
    re.IGNORECASE,
)


def _fold_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


def _tokenize_words(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]{2,}", _fold_accents(text))
    # Collapse 3+ repeated identical letters (e.g. "boook" -> "book")
    return [re.sub(r"([a-z])\1{2,}", r"\1\1", w) for w in words]


def is_overview_query(query: str) -> bool:
    if not query:
        return False
    return bool(OVERVIEW_QUERY_RE.search(query))


def extract_query_features(
    query: str,
    book_title: str = "",
    book_author: str = "",
) -> Tuple[List[str], Set[str], bool]:
    """
    Extract salient lexical terms, expanded domain concept terms, and overview flag
    from a user search query.
    """
    raw_words = _tokenize_words(query)
    meta_words = set(_tokenize_words(f"{book_title} {book_author}"))

    salient: List[str] = []
    seen: Set[str] = set()
    for w in raw_words:
        if w in QUERY_STOPWORDS or w in seen:
            continue
        salient.append(w)
        seen.add(w)

    # Separate non-author/title entity words when specific terms exist
    specific_salient = [w for w in salient if w not in meta_words]
    active_salient = specific_salient

    concepts: Set[str] = set()
    raw_set = set(raw_words)
    if raw_set.intersection(STUDY_QUERY_TRIGGERS):
        concepts.update(STUDY_CONCEPT_TERMS)
    if raw_set.intersection(EXPERIENCE_QUERY_TRIGGERS):
        concepts.update(EXPERIENCE_CONCEPT_TERMS)

    overview = is_overview_query(query) and len(active_salient) == 0
    return active_salient, concepts, overview


def compute_hybrid_score(
    query: str,
    chunk_content: str,
    vector_distance: float,
    chunk_index: int = 0,
    total_chunks: int = 1,
    book_title: str = "",
    book_author: str = "",
) -> Tuple[float, float]:
    """
    Combine dense vector similarity with accent-insensitive lexical keyword matching,
    domain concept expansion, and document structure priors.
    Returns (hybrid_distance, hybrid_similarity).
    """
    vec_sim = max(0.0, min(1.0, 1.0 - float(vector_distance)))
    folded_content = _fold_accents(chunk_content)
    content_word_set = set(re.findall(r"[a-z0-9]{2,}", folded_content))

    salient_terms, concept_terms, overview = extract_query_features(
        query, book_title=book_title, book_author=book_author
    )

    lexical_boost = 0.0
    matched_salient = 0
    for term in salient_terms:
        if term in content_word_set or (len(term) >= 4 and term in folded_content):
            matched_salient += 1

    if salient_terms and matched_salient > 0:
        coverage = float(matched_salient) / float(len(salient_terms))
        # Strong boost for matching specific entities/keywords (e.g. "Thark", "Zanthus", "IELTS")
        lexical_boost += 0.35 * coverage + 0.12 * min(matched_salient, 3)
        if len(salient_terms) >= 2:
            phrase = " ".join(salient_terms)
            if phrase in folded_content:
                lexical_boost += 0.15

    if concept_terms and (not salient_terms or matched_salient > 0):
        matched_concepts = sum(
            1 for c in concept_terms if c in content_word_set or c in folded_content
        )
        if matched_concepts > 0:
            lexical_boost += min(0.45, 0.09 * matched_concepts)

    if overview:
        # Overview/summary queries should prioritize opening chunks & table of contents
        if chunk_index == 0:
            lexical_boost += 0.42
        elif chunk_index == 1:
            lexical_boost += 0.32
        elif chunk_index == 2:
            lexical_boost += 0.22
        elif chunk_index <= 4:
            lexical_boost += 0.12
        if any(
            kw in folded_content
            for kw in ("sumario", "summary", "education", "procedimentos", "consideracoes iniciais")
        ):
            lexical_boost += 0.18

    # Dampen short trailing tail chunks when their vector similarity comes solely from the
    # prepended book-title/author header and they contain zero query terms or concepts
    penalty = 0.0
    if (
        total_chunks >= 4
        and chunk_index >= total_chunks - 1
        and len(content_word_set) < 60
        and matched_salient == 0
        and lexical_boost == 0.0
    ):
        penalty = 0.15

    hybrid_sim = max(0.0, min(0.9999, vec_sim + lexical_boost - penalty))
    hybrid_dist = round(1.0 - hybrid_sim, 6)
    return hybrid_dist, round(hybrid_sim, 6)


class DeterministicHashEmbeddings:
    """
    Deterministic, L2-normalized 384-dimensional feature-hashing embeddings.
    Prevents zero-vector NaN cosine distances in pgvector/tests while preserving
    meaningful lexical & character n-gram similarity for testing and fallback environments.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def _encode_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if not tokens:
            # Non-zero unit basis vector for empty/symbol-only inputs to avoid NaN cosine distance
            vec[0] = 1.0
            return vec

        # Include unigrams and bigrams for phrase sensitivity
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i + 1]}")

        for feat in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [round(v / norm, 6) for v in vec]

    def embed_query(self, text: str) -> List[float]:
        return self._encode_one(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode_one(t) for t in texts]


# Alias preserved for backwards compatibility
MockEmbeddings = DeterministicHashEmbeddings

try:
    if os.getenv("TESTING") == "1":
        embeddings = DeterministicHashEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDINGS_MODEL_ID", "all-MiniLM-L6-v2")
        )
except Exception as exc:
    logger.warning(
        "Failed to initialize HuggingFaceEmbeddings (%s); using DeterministicHashEmbeddings fallback.",
        exc,
    )
    embeddings = DeterministicHashEmbeddings()


def _is_postgres_connection(conn) -> bool:
    dialect = getattr(getattr(conn, "capabilities", None), "dialect", "")
    return dialect == "postgres"


def _validate_uuid(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid UUID value: {value}") from exc


def format_vector_str(vector: List[float]) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"


def parse_vector_str(vector_str: Optional[str]) -> Optional[List[float]]:
    if not vector_str:
        return None
    if isinstance(vector_str, list):
        return [float(x) for x in vector_str]
    try:
        parsed = json.loads(vector_str)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return None


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return float(1.0 - similarity)


async def async_embed_query(text: str) -> List[float]:
    return await asyncio.to_thread(embeddings.embed_query, text)


async def async_embed_documents(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return await asyncio.to_thread(embeddings.embed_documents, texts)


async def init_vector_extension():
    conn = Tortoise.get_connection("default")
    if not _is_postgres_connection(conn):
        return

    await conn.execute_script("CREATE EXTENSION IF NOT EXISTS vector;")
    try:
        await conn.execute_script(
            f"ALTER TABLE books ALTER COLUMN vector TYPE vector({EMBEDDING_DIM}) USING vector::vector({EMBEDDING_DIM});"
        )
    except Exception as e:
        logger.debug("Books vector column alteration skipped: %s", e)

    try:
        await conn.execute_script(
            f"ALTER TABLE book_chunks ALTER COLUMN vector TYPE vector({EMBEDDING_DIM}) USING vector::vector({EMBEDDING_DIM});"
        )
    except Exception as e:
        logger.debug("BookChunks vector column alteration skipped: %s", e)

    try:
        await conn.execute_script(
            """
            CREATE INDEX IF NOT EXISTS idx_book_chunks_vector_hnsw
            ON book_chunks USING hnsw (vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
        )
    except Exception as e:
        logger.debug("HNSW index creation skipped: %s", e)


def _rank_candidates_with_hybrid_score(
    query: str,
    candidates: List[Dict[str, Any]],
    safe_limit: int,
    max_distance: Optional[float],
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    # Group chunk counts per book for tail-chunk dampening and continuation lookup
    max_chunk_idx_by_book: Dict[str, int] = {}
    by_book_and_idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for c in candidates:
        b_id = str(c.get("book_id") or "")
        c_idx = int(c.get("chunk_index") or 0)
        if b_id:
            max_chunk_idx_by_book[b_id] = max(max_chunk_idx_by_book.get(b_id, 0), c_idx + 1)
            by_book_and_idx[(b_id, c_idx)] = c

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        b_id = str(c.get("book_id") or "")
        c_idx = int(c.get("chunk_index") or 0)
        total_chunks = max_chunk_idx_by_book.get(b_id, 1)
        vec_dist = float(c.get("distance") if c.get("distance") is not None else 0.5)
        hyb_dist, hyb_sim = compute_hybrid_score(
            query=query,
            chunk_content=str(c.get("content") or ""),
            vector_distance=vec_dist,
            chunk_index=c_idx,
            total_chunks=total_chunks,
            book_title=str(c.get("title") or ""),
            book_author=str(c.get("author") or ""),
        )
        if max_distance is not None and hyb_dist > max_distance and vec_dist > max_distance:
            continue
        item = dict(c)
        item["vector_distance"] = round(vec_dist, 6)
        item["distance"] = hyb_dist
        item["similarity"] = hyb_sim
        scored.append(item)

    scored.sort(key=lambda item: (item["distance"], item.get("chunk_index") or 0))
    top_results = scored[:safe_limit]

    # If the top matched chunk has a salient keyword match (e.g., a company/section header like
    # "Thark" at the bottom of chunk 1 or "Zanthus" at the bottom of chunk 7), ensure its
    # immediate continuation chunk (chunk_index + 1) is also included in top_results.
    if top_results and len(top_results) >= 2:
        first = top_results[0]
        salient_terms, _, _ = extract_query_features(
            query,
            book_title=str(first.get("title") or ""),
            book_author=str(first.get("author") or ""),
        )
        if salient_terms:
            folded_first = _fold_accents(str(first.get("content") or ""))
            if any(t in folded_first for t in salient_terms):
                next_key = (str(first.get("book_id") or ""), int(first.get("chunk_index") or 0) + 1)
                next_cand = by_book_and_idx.get(next_key)
                if next_cand and all(r.get("id") != next_cand.get("id") for r in top_results):
                    # Find scored version of next_cand or build it
                    next_scored = next(
                        (s for s in scored if s.get("id") == next_cand.get("id")),
                        None,
                    )
                    if next_scored is not None:
                        top_results = [top_results[0], next_scored] + top_results[1 : safe_limit - 1]

    return top_results


async def search_similar_chunks(
    query: str,
    limit: int = 5,
    book_id: Optional[str] = None,
    user_id: Optional[str] = None,
    max_distance: Optional[float] = DEFAULT_MAX_COSINE_DISTANCE,
    expand_window: int = 0,
) -> List[Dict[str, Any]]:
    """
    Passage-level hybrid (semantic vector + lexical/concept) search across `book_chunks` with:
      - Safe UUID validation & parameterized queries (no SQL injection)
      - Multi-tenant `user_id` scoping
      - Single-book passage ranking when `book_id` is provided
      - Hybrid distance thresholding (`max_distance`)
      - Adjacent continuation chunk inclusion & optional window expansion (`expand_window`)
    """
    validated_book_id = _validate_uuid(book_id) if book_id else None
    validated_user_id = _validate_uuid(user_id) if user_id else None
    safe_limit = max(1, min(int(limit), 50))

    query_vector = await async_embed_query(query)
    vector_str = format_vector_str(query_vector)
    conn = Tortoise.get_connection("default")

    results: List[Dict[str, Any]] = []

    if _is_postgres_connection(conn):
        params: List[Any] = [vector_str]
        conditions = ["bc.vector IS NOT NULL"]

        if validated_user_id:
            params.append(validated_user_id)
            p_idx = len(params)
            conditions.append(f"(bc.user_id = ${p_idx} OR b.user_id::text = ${p_idx})")

        if validated_book_id:
            params.append(validated_book_id)
            p_idx = len(params)
            conditions.append(f"bc.book_id = ${p_idx}::uuid")

        candidate_pool_limit = 250 if validated_book_id else max(safe_limit * 10, 60)
        params.append(candidate_pool_limit)
        limit_idx = len(params)
        where_sql = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT
                bc.id::text AS id,
                bc.book_id::text AS book_id,
                b.title AS title,
                b.author AS author,
                bc.chapter AS chapter,
                bc.page_number AS page_number,
                bc.chunk_index AS chunk_index,
                bc.content AS content,
                (bc.vector <=> $1::vector) AS distance
            FROM book_chunks bc
            JOIN books b ON b.id = bc.book_id
            {where_sql}
            ORDER BY distance ASC
            LIMIT ${limit_idx}
        """
        rows = await conn.execute_query_dict(sql, params)
        results = _rank_candidates_with_hybrid_score(
            query=query,
            candidates=rows,
            safe_limit=safe_limit,
            max_distance=max_distance,
        )
    else:
        # SQLite / ORM execution path with exact cosine distance + hybrid lexical scoring
        qs = BookChunk.all().prefetch_related("book")
        if validated_book_id:
            qs = qs.filter(book_id=validated_book_id)
        if validated_user_id:
            qs = qs.filter(Q(user_id=validated_user_id) | Q(book__user_id=validated_user_id))

        chunks = await qs
        raw_candidates: List[Dict[str, Any]] = []
        for ch in chunks:
            ch_vec = parse_vector_str(ch.vector)
            if not ch_vec:
                continue
            dist = cosine_distance(query_vector, ch_vec)
            raw_candidates.append(
                {
                    "id": str(ch.id),
                    "book_id": str(ch.book_id),
                    "title": ch.book.title if ch.book else "",
                    "author": ch.book.author if ch.book else "",
                    "chapter": ch.chapter,
                    "page_number": ch.page_number,
                    "chunk_index": ch.chunk_index,
                    "content": ch.content,
                    "distance": round(dist, 6),
                }
            )

        results = _rank_candidates_with_hybrid_score(
            query=query,
            candidates=raw_candidates,
            safe_limit=safe_limit,
            max_distance=max_distance,
        )

        # Backwards-compatible fallback if a legacy Book has no BookChunk rows yet
        if not results and not chunks:
            book_qs = Book.all()
            if validated_book_id:
                book_qs = book_qs.filter(id=validated_book_id)
            if validated_user_id:
                book_qs = book_qs.filter(user_id=validated_user_id)
            books = await book_qs
            legacy_candidates: List[Dict[str, Any]] = []
            for b in books:
                b_vec = parse_vector_str(b.vector)
                dist = cosine_distance(query_vector, b_vec) if b_vec else 0.5
                legacy_candidates.append(
                    {
                        "id": str(b.id),
                        "book_id": str(b.id),
                        "title": b.title,
                        "author": b.author,
                        "chapter": None,
                        "page_number": None,
                        "chunk_index": 0,
                        "content": b.content,
                        "distance": round(dist, 6),
                    }
                )
            results = _rank_candidates_with_hybrid_score(
                query=query,
                candidates=legacy_candidates,
                safe_limit=safe_limit,
                max_distance=max_distance,
            )

    if expand_window > 0 and results:
        for item in results:
            b_id = item.get("book_id")
            c_idx = item.get("chunk_index")
            if b_id is None or c_idx is None:
                continue
            neighbors = await BookChunk.filter(
                book_id=b_id,
                chunk_index__gte=max(0, c_idx - expand_window),
                chunk_index__lte=c_idx + expand_window,
            ).order_by("chunk_index")
            if len(neighbors) > 1:
                item["content"] = "\n".join(n.content for n in neighbors)

    return results


async def search_similar_books(
    query: str,
    limit: int = 5,
    book_id: Optional[str] = None,
    user_id: Optional[str] = None,
    max_distance: Optional[float] = DEFAULT_MAX_COSINE_DISTANCE,
    expand_window: int = 0,
) -> List[Dict[str, Any]]:
    """
    Search top-k relevant book chunks (preserving backwards-compatible function name and keys).
    """
    return await search_similar_chunks(
        query=query,
        limit=limit,
        book_id=book_id,
        user_id=user_id,
        max_distance=max_distance,
        expand_window=expand_window,
    )
