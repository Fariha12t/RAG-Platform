"""Sentence-aware chunking with overlap. Chunks never cross page boundaries, so every
chunk carries an exact page (or section) reference for citations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.ingestion.loaders import Page


@dataclass
class ChunkData:
    page: int
    index: int
    text: str
    heading: Optional[str] = None
    ocr: bool = False


def _units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:            # pathological long run without punctuation
                units.append(sent[:size])
                sent = sent[size:]
            if sent:
                units.append(sent)
    return units


def split_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    units = _units(text, size)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        if cur and cur_len + len(u) + 1 > size:
            chunks.append(" ".join(cur))
            keep: list[str] = []
            total = 0
            for prev in reversed(cur):          # carry trailing sentences as overlap
                if total + len(prev) + 1 > overlap:
                    break
                keep.insert(0, prev)
                total += len(prev) + 1
            cur, cur_len = keep, total
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        final = " ".join(cur)
        if not chunks or final != chunks[-1]:
            chunks.append(final)
    return chunks


def chunk_pages(pages: list[Page], size: int = 900, overlap: int = 150) -> list[ChunkData]:
    out: list[ChunkData] = []
    for page in pages:
        if not page.text.strip():
            continue
        for piece in split_text(page.text, size, overlap):
            out.append(ChunkData(page=page.number, index=len(out), text=piece,
                                 heading=page.heading, ocr=page.ocr))
    return out
