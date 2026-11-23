"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion, then reranked."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.retrieval.embeddings import get_embedder
from app.retrieval.index import get_index
from app.retrieval.reranker import get_reranker

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Retrieved:
    chunk_id: int
    document_id: int
    filename: str
    page: int
    heading: Optional[str]
    text: str
    score: float
    flagged: bool = False
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return fused


def search(db: Session, collection_id: int, query: str, top_k: Optional[int] = None,
           mode: str = "hybrid_rerank", candidate_k: Optional[int] = None) -> list[Retrieved]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = get_settings()
    top_k = top_k or s.top_k
    candidate_k = max(candidate_k or s.candidate_k, top_k)
    index = get_index(db, collection_id)
    if not index.chunk_ids:
        return []

    dense = index.dense(get_embedder().embed_query(query), candidate_k) if mode != "bm25" else []
    keyword = index.keyword(query, candidate_k) if mode != "dense" else []
    dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
    bm25_rank = {i: r for r, (i, _) in enumerate(keyword, 1)}

    if mode == "dense":
        ordered = [(i, sc) for i, sc in dense]
    elif mode == "bm25":
        ordered = [(i, sc) for i, sc in keyword]
    else:
        fused = rrf([[i for i, _ in dense], [i for i, _ in keyword]])
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])

    rerank_scores: dict[int, float] = {}
    reranker = get_reranker() if mode == "hybrid_rerank" else None
    if reranker is not None and ordered:
        cand = ordered[:candidate_k]
        scores = reranker.score(query, [index.meta[i]["text"] for i, _ in cand], index.bm25)
        # blend rerank score with fused rank so ties are broken by first-stage retrieval
        n = len(cand)
        blended = {i: sc + 0.05 * (1 - pos / n) for pos, ((i, _), sc) in enumerate(zip(cand, scores))}
        rerank_scores = {i: sc for (i, _), sc in zip(cand, scores)}
        ordered = sorted(blended.items(), key=lambda kv: -kv[1])

    results = []
    for i, sc in ordered[:top_k]:
        m = index.meta[i]
        results.append(Retrieved(
            chunk_id=m["chunk_id"], document_id=m["document_id"], filename=m["filename"],
            page=m["page"], heading=m["heading"], text=m["text"], score=float(sc),
            flagged=m["flagged"], dense_rank=dense_rank.get(i), bm25_rank=bm25_rank.get(i),
            rerank_score=rerank_scores.get(i)))
    return results
