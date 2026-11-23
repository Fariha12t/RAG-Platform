"""In-memory per-collection index (dense matrix + BM25) built from chunks stored in the DB.
The DB is the source of truth; the index is rebuilt lazily after ingest/delete/re-index.
(For multi-worker deployments swap this for pgvector / a shared vector DB.)"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Chunk, Document
from app.retrieval.bm25 import BM25
from app.textutils import tokenize


@dataclass
class CollectionIndex:
    chunk_ids: list[int]
    meta: list[dict]
    matrix: np.ndarray
    bm25: BM25
    faiss_index: object = None

    def dense(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.chunk_ids):
            return []
        k = min(k, len(self.chunk_ids))
        if self.faiss_index is not None:
            scores, idx = self.faiss_index.search(qvec.reshape(1, -1).astype(np.float32), k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        sims = self.matrix @ qvec
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def keyword(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.scores(tokenize(query))
        if not scores.size:
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


_cache: dict[int, CollectionIndex] = {}
_lock = threading.Lock()


def invalidate(collection_id: int) -> None:
    with _lock:
        _cache.pop(collection_id, None)


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def get_index(db: Session, collection_id: int) -> CollectionIndex:
    with _lock:
        if collection_id in _cache:
            return _cache[collection_id]
    rows = db.execute(
        select(Chunk, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.collection_id == collection_id, Document.status == "ready")
        .order_by(Chunk.id)
    ).all()
    ids, meta, vecs, toks = [], [], [], []
    for chunk, filename in rows:
        ids.append(chunk.id)
        meta.append({"chunk_id": chunk.id, "document_id": chunk.document_id, "filename": filename,
                     "page": chunk.page, "heading": chunk.heading, "text": chunk.text,
                     "flagged": chunk.flagged})
        vecs.append(np.frombuffer(chunk.embedding, dtype=np.float32))
        toks.append(tokenize(chunk.text))
    matrix = np.vstack(vecs) if vecs else np.zeros((0, 1), dtype=np.float32)
    faiss_index = None
    if vecs and get_settings().vector_backend == "faiss":
        try:
            import faiss
            faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            faiss_index.add(np.ascontiguousarray(matrix))
        except ImportError:
            faiss_index = None
    idx = CollectionIndex(ids, meta, matrix, BM25(toks), faiss_index)
    with _lock:
        _cache[collection_id] = idx
    return idx
