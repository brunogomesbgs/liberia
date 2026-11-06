import asyncio
import os
import re
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv(override=False)

from app.rag import (
    search_similar_books,
    extract_query_features,
    is_overview_query,
    _fold_accents,
)
from app.ingestion import estimate_token_count

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "480"))
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# Match explicit escalation requests without false-positives on "humanity", "human nature", "helpful", etc.
ESCALATION_PATTERNS = re.compile(
    r"(?:^\s*(?:human|help|support|admin|operator|live\s+agent)\s*[.!?]*$)"
    r"|(?:\b(?:talk|speak|connect|transfer|switch|chat)\s+(?:to\s+|with\s+)?(?:a\s+|an\s+)?(?:human|person|admin|agent|representative|operator|support)\b)"
    r"|(?:\b(?:need|want|request|require|get)\s+(?:a\s+)?(?:human|live\s+agent|real\s+person|human\s+help|human\s+assistant|human\s+support)\b)"
    r"|(?:\b(?:i\s+need\s+help|help\s+me|human\s+assistant|human\s+agent|human\s+support|customer\s+support|escalate)\b)",
    re.IGNORECASE,
)

# Anaphoric pronouns/markers that genuinely refer to the previous conversation turn
# (excludes self-contained document references like "this book", "this document", "it is")
FOLLOWUP_PRONOUN_RE = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|next|then|after\s+that|why\s+did|what\s+else|tell\s+me\s+more)\b",
    re.IGNORECASE,
)

YES_NO_QUESTION_RE = re.compile(
    r"^(?:did|does|do|has|have|had|is|are|was|were|can|could|would|should)\b"
    r"|(?:^(?:the\s+author|he|she|bruno(?:\s+silva)?)\s+(?:has|have|had|worked|studied|is|was)\b)",
    re.IGNORECASE,
)


def is_escalation_request(message_text: str) -> bool:
    if not message_text:
        return False
    return bool(ESCALATION_PATTERNS.search(message_text.strip()))


def _clean_passage_for_context(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\.{3,}", " - ", text)
    cleaned = re.sub(r"[-_=]{4,}", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_contextual_search_query(messages: List[BaseMessage]) -> str:
    """
    Combine recent user conversation turns ONLY when the latest query is a genuine
    elliptical fragment (<= 2 words) or contains anaphoric pronouns without a new proper noun.
    Prevents standalone short questions (e.g., "did the author worked at Thark")
    from being polluted by previous unrelated questions (e.g., "did the author worked at Volkswagen").
    """
    if not messages:
        return ""
    last_content = (messages[-1].content or "").strip()
    if len(messages) < 2 or not last_content:
        return last_content

    words = last_content.split()
    word_count = len(words)
    # Check if the query introduces a capitalized entity after the first word
    has_new_entity = any(w[0].isupper() and len(w) > 1 for w in words[1:] if w and w[0].isalpha())
    if has_new_entity:
        return last_content

    needs_history = word_count <= 2 or bool(FOLLOWUP_PRONOUN_RE.search(last_content))
    if not needs_history:
        return last_content

    prior_user_msgs = [
        (m.content or "").strip()
        for m in messages[:-1]
        if isinstance(m, HumanMessage) and (m.content or "").strip()
    ]
    if not prior_user_msgs:
        return last_content

    prev_turn = prior_user_msgs[-1]
    return f"{prev_turn}\n{last_content}"


def build_bounded_context(
    docs: List[Dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """
    Assemble retrieved passages into a clean prompt context bounded by `max_tokens`.
    Avoids bracketed `[Book: ...]` headers on every chunk that cause extractive T5 models
    to echo metadata headers instead of passage content.
    """
    selected: List[str] = []
    used_tokens = 0

    for d in docs:
        raw_content = _clean_passage_for_context(str(d.get("content") or ""))
        if not raw_content:
            continue

        snippet = raw_content
        snippet_tokens = estimate_token_count(snippet)
        if selected and (used_tokens + snippet_tokens > max_tokens):
            # If there is still room for at least 60 tokens, include a truncated prefix of this chunk
            remaining = max_tokens - used_tokens
            if remaining >= 60:
                words = snippet.split()
                max_words = max(30, int(remaining / 1.15))
                snippet = " ".join(words[:max_words])
                selected.append(snippet)
            break

        if not selected and snippet_tokens > max_tokens:
            words = snippet.split()
            max_words = max(20, int(max_tokens / 1.15))
            snippet = " ".join(words[:max_words])
            snippet_tokens = estimate_token_count(snippet)

        selected.append(snippet)
        used_tokens += snippet_tokens

    return "\n\n".join(selected)


def _extract_supporting_lines(
    question: str,
    docs: List[Dict[str, Any]],
    max_chars: int = 420,
) -> str:
    """
    Extract the most relevant lines/sentences from retrieved `docs` that match the
    query's salient terms or expanded domain concepts, preserving original UTF-8 text.
    """
    if not docs:
        return ""

    first_doc = docs[0]
    book_title = str(first_doc.get("title") or "")
    book_author = str(first_doc.get("author") or "")
    salient_terms, concept_terms, overview = extract_query_features(
        question, book_title=book_title, book_author=book_author
    )

    if overview:
        ordered = sorted(docs, key=lambda d: int(d.get("chunk_index") or 0))
        combined_lines: List[str] = []
        for d in ordered[:3]:
            cleaned = _clean_passage_for_context(str(d.get("content") or ""))
            for line in cleaned.split("\n"):
                s = line.strip()
                if s and len(s) > 2 and s not in combined_lines:
                    combined_lines.append(s)
        summary_text = "; ".join(combined_lines[:10])
        return summary_text[:max_chars].strip()

    # Score logical lines/blocks across top retrieved chunks
    all_lines: List[str] = []
    for d in docs:
        cleaned = _clean_passage_for_context(str(d.get("content") or ""))
        for line in cleaned.split("\n"):
            s = line.strip()
            if s:
                all_lines.append(s)

    if not all_lines:
        return ""

    best_idx = 0
    best_score = -1.0
    for idx, line in enumerate(all_lines):
        folded = _fold_accents(line)
        words = set(re.findall(r"[a-z0-9]{2,}", folded))
        score = 0.0
        matched_salient_on_line = 0
        for t in salient_terms:
            if t in words or (len(t) >= 4 and t in folded):
                score += 10.0
                matched_salient_on_line += 1
        if not salient_terms or matched_salient_on_line > 0:
            for c in concept_terms:
                if c in words or c in folded:
                    score += 1.5
        # Bonus if neighboring lines also match
        if idx + 1 < len(all_lines):
            next_folded = _fold_accents(all_lines[idx + 1])
            for t in salient_terms:
                if t in next_folded:
                    score += 2.0
            for c in concept_terms:
                if c in next_folded:
                    score += 0.5
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score <= 0:
        return " ".join(all_lines[:4])[:max_chars].strip()

    # Collect a window starting at best_idx (skipping solitary bullet numbers like "1.", "2.")
    window_start = best_idx
    collected: List[str] = []
    total_len = 0
    for line in all_lines[window_start : window_start + 8]:
        if re.match(r"^(?:\d+\.|[•\-*])\s*$", line):
            continue
        if line not in collected:
            collected.append(line)
            total_len += len(line) + 2
            if total_len >= max_chars:
                break

    snippet = " ".join(collected).strip()
    return snippet[:max_chars].strip()


def synthesize_grounded_answer(
    question: str,
    raw_answer: str,
    docs: List[Dict[str, Any]],
) -> str:
    """
    Combine the LLM's generated answer with grounded evidence verification over the
    retrieved chunks so that:
      - Yes/No questions (e.g. "did the author worked at Thark" vs "Volkswagen") are
        verified against the retrieved document text and include supporting details.
      - Terse 1-word T5 extractions (e.g. "Uninove") or `<unk>`-stripped outputs are
        enriched with the exact UTF-8 supporting lines from the document.
      - Overview/summary requests return a structured summary of the document.
    """
    cleaned_raw = (raw_answer or "").strip()
    q_stripped = (question or "").strip()
    if not docs:
        if bool(YES_NO_QUESTION_RE.search(q_stripped)):
            return "No. That is not mentioned in the document."
        return cleaned_raw or "No relevant passages were found in the book to answer this question."

    # Preserve pure MockLLM output only when running basic generic greetings without query substance
    if cleaned_raw.startswith("[Mock]") and len(q_stripped.split()) <= 2:
        return cleaned_raw

    first_doc = docs[0]
    book_title = str(first_doc.get("title") or "the document").strip()
    book_author = str(first_doc.get("author") or "").strip()
    salient_terms, concept_terms, overview = extract_query_features(
        q_stripped, book_title=book_title, book_author=book_author
    )

    combined_folded = _fold_accents(" ".join(str(d.get("content") or "") for d in docs))
    combined_words = set(re.findall(r"[a-z0-9]{2,}", combined_folded))

    matched_salient = [
        t for t in salient_terms if t in combined_words or (len(t) >= 4 and t in combined_folded)
    ]
    matched_concepts = [
        c for c in concept_terms if c in combined_words or c in combined_folded
    ]
    evidence = _extract_supporting_lines(q_stripped, docs)

    # 1. Overview / summary / "make a resume" queries
    if overview:
        header_desc = f"{book_title}" + (f" by {book_author}" if book_author and book_author.lower() not in book_title.lower() else "")
        if evidence:
            return f"Summary of {header_desc}: {evidence}"
        return cleaned_raw

    # 2. Yes / No & verification questions (e.g. "did the author worked at Thark", "did the author worked at Volkswagen")
    is_yn = bool(YES_NO_QUESTION_RE.search(q_stripped))
    raw_lower = cleaned_raw.lower().strip(".,!?:; ")

    if is_yn:
        has_salient_evidence = (
            len(salient_terms) > 0 and len(matched_salient) == len(salient_terms)
        ) or (
            len(salient_terms) == 0 and len(matched_concepts) > 0
        )

        if has_salient_evidence:
            if evidence:
                return f"Yes. According to {book_title}: {evidence}"
            return "Yes."

        # If the user asked about a specific entity (e.g. "Volkswagen", "Redraw") that does NOT appear in the book
        if salient_terms and len(matched_salient) == 0:
            missing_terms = ", ".join(salient_terms)
            return f"No. There is no mention of {missing_terms} in {book_title}."

        if raw_lower in ("yes", "true", "sim"):
            return f"Yes. According to {book_title}: {evidence}" if evidence else "Yes."
        if raw_lower in ("no", "false", "nao", "não"):
            return f"No. That is not mentioned in {book_title}."

    # 3. Factual / Wh- questions (e.g. "where Bruno Silva studied", "what is said about Energia Elétrica")
    folded_raw = _fold_accents(cleaned_raw)
    has_expected_focus = True
    if salient_terms or concept_terms:
        has_expected_focus = any(t in folded_raw for t in salient_terms) or any(
            c in folded_raw for c in concept_terms
        )

    # Detect if raw_answer is unusable (echoed bracket header, bare "no" on open question, off-topic span, or Mock)
    is_bad_raw = (
        not cleaned_raw
        or cleaned_raw.startswith("[Book:")
        or cleaned_raw.startswith("[Mock]")
        or raw_lower in ("no", "yes", "a resume", "resume", "3", "1", "2")
        or not has_expected_focus
    )
    if is_bad_raw:
        if evidence:
            return evidence
        return cleaned_raw

    # If the LLM gave a short span (e.g. "Uninove"), enrich it with the supporting lines from the document
    if len(cleaned_raw.split()) <= 6 and evidence:
        if folded_raw in _fold_accents(evidence):
            return f"{cleaned_raw} — {evidence}"
        return evidence

    if evidence:
        return evidence
    return cleaned_raw


# Define the state for the graph
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    context: str
    book_id: Optional[str]
    user_id: Optional[str]
    human_required: bool
    retrieved_chunk_ids: List[str]
    retrieved_docs: List[Dict[str, Any]]
    top_similarity_score: Optional[float]


class MockLLM:
    def invoke(self, messages, **kwargs):
        return AIMessage(content="[Mock] This is a free assistant response.")

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)


# Using a mock or conditional loading for tests to avoid heavy downloads
try:
    if os.getenv("TESTING") == "1":
        llm = MockLLM()
    else:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from langchain_core.language_models.llms import LLM
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun

        model_id = os.getenv("LLM_MODEL_ID", "google/flan-t5-large")
        print(f"Loading model: {model_id}...")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = T5ForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True)
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully on {DEVICE}.")

        class LocalT5LLM(LLM):
            def _call(
                self,
                prompt: str,
                stop: Optional[List] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> str:
                with torch.inference_mode():
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        no_repeat_ngram_size=3,
                    )
                    return tokenizer.decode(outputs[0], skip_special_tokens=True)

            def invoke(self, messages, **kwargs):
                if isinstance(messages, list):
                    prompt = ""
                    for m in messages:
                        if isinstance(m, BaseMessage):
                            prompt += f"{m.content}\n"
                        else:
                            prompt += f"{str(m)}\n"
                else:
                    prompt = str(messages)
                content = self._call(prompt, **kwargs)
                return AIMessage(content=content)

            async def ainvoke(self, messages, **kwargs):
                return await asyncio.to_thread(self.invoke, messages, **kwargs)

            @property
            def _llm_type(self) -> str:
                return "local_t5"

        llm = LocalT5LLM()
except Exception as e:
    import traceback

    print(f"CRITICAL: Error loading model: {e}. Falling back to a placeholder.")
    traceback.print_exc()

    class FallbackMockLLM:
        def invoke(self, messages, **kwargs):
            return AIMessage(content=f"[Error Fallback] {str(e)[:100]}")

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    llm = FallbackMockLLM()


def route_intent(state: AgentState) -> str:
    """Route escalation requests BEFORE retrieval to avoid wasted embedding/DB calls."""
    messages = state.get("messages") or []
    last_message = messages[-1].content if messages else ""
    if is_escalation_request(last_message):
        return "escalate"
    return "retrieve"


async def escalate(state: AgentState):
    return {
        "messages": [AIMessage(content="I'm connecting you with a human assistant. Please wait.")],
        "human_required": True,
        "retrieved_chunk_ids": [],
        "retrieved_docs": [],
        "top_similarity_score": None,
    }


async def retrieve(state: AgentState):
    messages = state.get("messages") or []
    search_query = build_contextual_search_query(messages)
    book_id = state.get("book_id")
    user_id = state.get("user_id")

    docs = await search_similar_books(
        search_query,
        limit=DEFAULT_RETRIEVAL_TOP_K,
        book_id=book_id,
        user_id=user_id,
    )
    context = build_bounded_context(docs, max_tokens=MAX_CONTEXT_TOKENS)
    chunk_ids = [str(d["id"]) for d in docs if d.get("id")]
    top_sim = (
        max((float(d["similarity"]) for d in docs if d.get("similarity") is not None), default=None)
        if docs
        else None
    )
    print(f"RAG Context found for book_id {book_id} ({len(docs)} chunks): {context[:100]}...")
    return {
        "context": context,
        "retrieved_chunk_ids": chunk_ids,
        "retrieved_docs": docs,
        "top_similarity_score": top_sim,
    }


async def assistant(state: AgentState):
    messages_list = state.get("messages") or []
    last_message = messages_list[-1].content if messages_list else ""
    if is_escalation_request(last_message):
        return await escalate(state)

    context = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs") or []
    contextual_question = build_contextual_search_query(messages_list) or last_message

    if context:
        prompt_text = (
            "Answer the question concisely based strictly on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {contextual_question}\n"
            "Answer:"
        )
    else:
        prompt_text = (
            "You are a book assistant. No relevant book passages matched the query; "
            "answer helpful general questions or state if the detail is not in the book.\n\n"
            f"Question: {contextual_question}\n"
            "Answer:"
        )

    messages = [HumanMessage(content=prompt_text)]

    if hasattr(llm, "ainvoke"):
        res = await llm.ainvoke(messages)
    else:
        res = await asyncio.to_thread(llm.invoke, messages)

    raw_content = res.content if hasattr(res, "content") else str(res)
    final_content = synthesize_grounded_answer(
        question=contextual_question,
        raw_answer=raw_content,
        docs=retrieved_docs,
    )
    return {
        "messages": [AIMessage(content=final_content)],
        "human_required": False,
        "retrieved_chunk_ids": state.get("retrieved_chunk_ids", []),
        "retrieved_docs": retrieved_docs,
        "top_similarity_score": state.get("top_similarity_score"),
    }


# Build the graph with pre-retrieval conditional routing
workflow = StateGraph(AgentState)
workflow.add_node("escalate", escalate)
workflow.add_node("retrieve", retrieve)
workflow.add_node("assistant", assistant)
workflow.add_conditional_edges(
    START,
    route_intent,
    {
        "escalate": "escalate",
        "retrieve": "retrieve",
    },
)
workflow.add_edge("escalate", END)
workflow.add_edge("retrieve", "assistant")
workflow.add_edge("assistant", END)

agent_app = workflow.compile()
