"""RAG orchestration: retrieve -> sanitise -> prompt -> generate -> parse citations -> persist."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Conversation, Message
from app.generation.guardrails import sanitize_source
from app.generation.llm import NOT_FOUND, get_llm
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.hybrid import Retrieved, search
from app.services.costs import log_usage

CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class ChatResult:
    conversation_id: int
    answer: str
    citations: list[dict]
    sources: list[dict]
    usage: dict
    injection_redactions: int


def parse_citations(answer: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def prepare_sources(hits: list[Retrieved]) -> tuple[list[dict], int]:
    sources, redactions = [], 0
    for n, h in enumerate(hits, start=1):
        clean, modified = sanitize_source(h.text)
        redactions += int(modified)
        sources.append({"n": n, "chunk_id": h.chunk_id, "document_id": h.document_id,
                        "filename": h.filename, "page": h.page, "heading": h.heading,
                        "text": clean, "score": round(h.score, 4), "flagged": h.flagged})
    return sources, redactions


def answer_question(question: str, hits: list[Retrieved], history: Optional[list[dict]] = None):
    """Pure generation step (no DB): returns (answer, citations, sources, llm_result, redactions)."""
    sources, redactions = prepare_sources(hits)
    llm = get_llm()
    if not sources:
        from app.generation.llm import LLMResult
        res = LLMResult(NOT_FOUND, 0, 0, getattr(llm, "model", "none"))
        return NOT_FOUND, [], sources, res, 0
    messages = [*(history or []), {"role": "user", "content": build_user_prompt(question, sources)}]
    res = llm.generate(SYSTEM_PROMPT, messages, sources=sources, question=question)
    cited = parse_citations(res.text, len(sources))
    citations = [{"n": n, **{k: sources[n - 1][k] for k in ("document_id", "filename", "page", "heading")},
                  "snippet": sources[n - 1]["text"][:300]} for n in cited]
    return res.text, citations, sources, res, redactions


def chat(db: Session, user_id: int, collection_id: int, question: str,
         conversation_id: Optional[int] = None, top_k: Optional[int] = None,
         mode: str = "hybrid_rerank") -> ChatResult:
    conv = db.get(Conversation, conversation_id) if conversation_id else None
    if conv is not None and (conv.user_id != user_id or conv.collection_id != collection_id):
        raise PermissionError("Conversation does not belong to this user/collection")
    if conv is None:
        conv = Conversation(user_id=user_id, collection_id=collection_id, title=question[:80])
        db.add(conv)
        db.flush()

    prior = [m for m in conv.messages][-6:]
    history = [{"role": m.role, "content": m.content} for m in prior]
    retrieval_query = question
    last_user = next((m.content for m in reversed(prior) if m.role == "user"), None)
    if last_user and len(question.split()) < 6:       # short follow-up -> carry context
        retrieval_query = f"{last_user} {question}"

    hits = search(db, collection_id, retrieval_query, top_k=top_k or get_settings().top_k, mode=mode)
    answer, citations, sources, res, redactions = answer_question(question, hits, history)

    cost = log_usage(db, user_id, "chat", res.model, res.input_tokens, res.output_tokens)
    cited_ns = {c["n"] for c in citations}
    for s in sources:
        s["cited"] = s["n"] in cited_ns
    db.add(Message(conversation_id=conv.id, role="user", content=question))
    db.add(Message(conversation_id=conv.id, role="assistant", content=answer, citations=citations))
    db.flush()
    return ChatResult(conv.id, answer, citations, sources,
                      {"model": res.model, "input_tokens": res.input_tokens,
                       "output_tokens": res.output_tokens, "cost_usd": cost}, redactions)
