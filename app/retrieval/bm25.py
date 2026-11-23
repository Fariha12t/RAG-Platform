"""Minimal BM25 (Okapi) keyword index with an inverted index."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return s
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s
