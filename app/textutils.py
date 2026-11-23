"""Shared tokenisation used by BM25, the hashing embedder, reranker and evaluation."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""a an and are as at be been being by can could did do does for from had has have how i in into is it its
of on or our over per shall should so than that the their them then there these this those to was were what when where which
who whom whose why will with would you your we us not no if but also any all each more most other such via""".split())

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(tok: str) -> str:
    if tok.isdigit() or any(c.isdigit() for c in tok):
        return tok
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    text = _NUM_COMMA.sub("", text.lower())
    toks = _TOKEN.findall(text)
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]
