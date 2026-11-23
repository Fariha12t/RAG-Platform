"""Rerankers. `lexical` is dependency-free (IDF-weighted query coverage + phrase match);
`cross-encoder` uses a Sentence Transformers CrossEncoder for much better relevance."""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.retrieval.bm25 import BM25
from app.textutils import tokenize


class LexicalReranker:
    name = "lexical"

    def score(self, query: str, texts: list[str], bm25: BM25) -> list[float]:
        q = tokenize(query)
        q_set = list(dict.fromkeys(q))
        if not q_set:
            return [0.0] * len(texts)
        idf = {t: bm25.idf(t) for t in q_set}
        total = sum(idf.values()) or 1.0
        q_bigrams = set(zip(q, q[1:]))
        out = []
        for text in texts:
            toks = tokenize(text)
            tset = set(toks)
            coverage = sum(idf[t] for t in q_set if t in tset) / total
            bigrams = set(zip(toks, toks[1:]))
            phrase = len(q_bigrams & bigrams) / len(q_bigrams) if q_bigrams else 0.0
            out.append(0.75 * coverage + 0.25 * phrase)
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.name = model_name

    def score(self, query: str, texts: list[str], bm25: BM25) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


@lru_cache(maxsize=1)
def get_reranker():
    s = get_settings()
    if s.reranker_backend == "cross-encoder":
        return CrossEncoderReranker(s.reranker_model)
    if s.reranker_backend == "none":
        return None
    return LexicalReranker()
