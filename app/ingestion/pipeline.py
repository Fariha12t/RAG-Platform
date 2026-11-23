"""Ingestion pipeline: save file -> extract (OCR fallback) -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

import numpy as np
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Chunk, Collection, Document
from app.generation.guardrails import scan
from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.retrieval.embeddings import get_embedder
from app.retrieval.index import invalidate
from app.services.costs import estimate_tokens, log_usage

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:200] or "file"


def ingest_file(db: Session, collection: Collection, src: Path, filename: str,
                user_id: int) -> tuple[Document, bool]:
    """Returns (document, created). Duplicate uploads (same SHA-256) are skipped."""
    s = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    if src.stat().st_size > s.max_upload_mb * 1024 * 1024:
        raise ValueError(f"File exceeds {s.max_upload_mb} MB limit")
    sha = _sha256(src)
    existing = db.query(Document).filter_by(collection_id=collection.id, sha256=sha).first()
    if existing:
        return existing, False
    dest_dir = s.upload_dir / str(collection.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sha[:12]}_{_safe_name(filename)}"
    shutil.copyfile(src, dest)
    doc = Document(collection_id=collection.id, filename=Path(filename).name, file_path=str(dest),
                   sha256=sha, uploaded_by=user_id, status="processing")
    db.add(doc)
    db.flush()
    index_document(db, doc, user_id)
    return doc, True


def index_document(db: Session, doc: Document, user_id: int) -> Document:
    """(Re)build chunks and embeddings for a stored document."""
    s = get_settings()
    db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    doc.status, doc.error = "processing", None
    try:
        pages = load_document(Path(doc.file_path))
        chunks = chunk_pages(pages, s.chunk_size, s.chunk_overlap)
        if not chunks:
            raise ValueError("No extractable text found (is OCR installed for scanned files?)")
        embedder = get_embedder()
        vectors = embedder.embed_documents([c.text for c in chunks]).astype(np.float32)
        flagged = 0
        for c, vec in zip(chunks, vectors):
            is_flagged = bool(scan(c.text))
            flagged += is_flagged
            db.add(Chunk(document_id=doc.id, collection_id=doc.collection_id, page=c.page,
                         chunk_index=c.index, heading=c.heading, text=c.text,
                         embedding=vec.tobytes(), flagged=is_flagged))
        doc.n_pages = len(pages)
        doc.n_chunks = len(chunks)
        doc.used_ocr = any(p.ocr for p in pages)
        doc.flagged_chunks = flagged
        doc.embedding_model = embedder.name
        doc.status = "ready"
        log_usage(db, user_id, "embedding", embedder.name,
                  sum(estimate_tokens(c.text) for c in chunks), 0)
        if flagged:
            log.warning("%s: %d chunk(s) flagged for possible prompt injection", doc.filename, flagged)
    except Exception as exc:
        log.exception("Ingestion failed for %s", doc.filename)
        doc.status, doc.error = "failed", str(exc)[:500]
    invalidate(doc.collection_id)
    db.flush()
    return doc


def delete_document(db: Session, doc: Document) -> None:
    path = Path(doc.file_path)
    collection_id = doc.collection_id
    db.delete(doc)
    db.flush()
    path.unlink(missing_ok=True)
    invalidate(collection_id)
