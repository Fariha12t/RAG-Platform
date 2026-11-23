"""Embedding backends.

* hashing  - dependency-free, deterministic signed feature-hashing of unigrams+bigrams.
             Great for offline demos/tests/CI; behaves like a lexical vector model.
* sbert    - Sentence Transformers (e.g. all-MiniLM-L6-v2). Recommended for real use.
* openai   - OpenAI embeddings API.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import get_settings
from app.textutils import tokenize


class Embedder(Protocol):
    name: str
    dim: int
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats: Counter = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        v = np.zeros(self.dim, dtype=np.float32)
        for f, c in feats.items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            weight = (1.0 + math.log(c)) * (0.7 if "_" in f else 1.0)
            v[h % self.dim] += weight if (h >> 63) & 1 else -weight
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SBertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.name = model_name
        self.dim = 1536

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.name, input=texts[i:i + 64])
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "sbert":
        return SBertEmbedder(s.embedding_model)
    if s.embedding_backend == "openai":
        return OpenAIEmbedder()
    return HashingEmbedder(s.embedding_dim)
