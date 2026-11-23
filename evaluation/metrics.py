"""Retrieval and answer metrics. All functions are pure and unit-tested."""
from __future__ import annotations

import re
from typing import Iterable

from app.textutils import split_sentences, tokenize


def norm(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return " ".join(text.split())


def evidence_found(chunk_text: str, evidence: str) -> bool:
    return norm(evidence) in norm(chunk_text)


def retrieval_metrics(hits: list[dict], expected_docs: list[str], evidence: list[str], ks=(1, 3, 5)) -> dict:
    """hits: ranked list of {'filename','text'}.
    Recall@K  = fraction of required evidence snippets present in the top-K chunks (of expected docs).
    MRR       = 1 / rank of the first chunk containing any evidence snippet.
    DocHit@K  = 1 if any top-K chunk comes from an expected document."""
    out: dict = {}
    for k in ks:
        top = hits[:k]
        found = [any(h["filename"] in expected_docs and evidence_found(h["text"], e) for h in top)
                 for e in evidence]
        out[f"recall@{k}"] = sum(found) / len(evidence) if evidence else 0.0
        out[f"dochit@{k}"] = float(any(h["filename"] in expected_docs for h in top))
    rr = 0.0
    for rank, h in enumerate(hits, 1):
        if h["filename"] in expected_docs and any(evidence_found(h["text"], e) for e in evidence):
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


_CITE = re.compile(r"\[(\d+)\]")


def groundedness(answer: str, sources: list[dict], threshold: float = 0.8) -> float:
    """Heuristic faithfulness: a sentence is grounded if >= `threshold` of its content tokens occur in the
    source(s) it cites. Uncited sentences count as ungrounded. (Swap in an LLM judge for production.)"""
    by_n = {s["n"]: set(tokenize(s["text"])) for s in sources}
    # keep "[n]" markers attached to the sentence they follow: "text. [1]" -> "text[1]."
    text = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r"\2\1", answer.replace("\n", " "))
    sentences = [s for s in split_sentences(text) if s.strip("-• ")]
    if not sentences:
        return 0.0
    ok = 0
    for sent in sentences:
        cited = [int(n) for n in _CITE.findall(sent)]
        toks = tokenize(_CITE.sub("", sent))
        if not cited or not toks:
            continue
        support = set().union(*(by_n.get(n, set()) for n in cited))
        ok += (sum(t in support for t in toks) / len(toks)) >= threshold
    return ok / len(sentences)


def citation_accuracy(citations: list[dict], expected_docs: list[str]) -> float:
    """Share of cited sources that belong to an expected document."""
    if not citations:
        return 0.0
    return sum(c["filename"] in expected_docs for c in citations) / len(citations)


def keyword_accuracy(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    a = norm(answer)
    return sum(norm(k) in a for k in keywords) / len(keywords)
