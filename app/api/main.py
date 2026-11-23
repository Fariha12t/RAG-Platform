"""FastAPI REST API. Auth: send your key in the `X-API-Key` header."""
from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import schemas
from app.db import (Chunk, Collection, CollectionAccess, Conversation, Document, User, init_db,
                    session_scope)
from app.ingestion.pipeline import delete_document, index_document, ingest_file
from app.retrieval.hybrid import search
from app.retrieval.index import invalidate
from app.services import auth, rag
from app.services.costs import usage_summary


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with session_scope() as db:
        auth.bootstrap_admin(db)
    yield


app = FastAPI(title="Enterprise RAG Document Intelligence Platform", version="1.0.0", lifespan=lifespan)


# ----------------------------------------------------------------------------- dependencies
def get_db():
    with session_scope() as db:
        yield db


def current_user(x_api_key: Optional[str] = Header(default=None), db: Session = Depends(get_db)) -> User:
    user = auth.authenticate(db, x_api_key) if x_api_key else None
    if user is None:
        raise HTTPException(401, "Missing or invalid X-API-Key")
    return user


def _collection(db: Session, user: User, collection_id: int, write: bool = False) -> Collection:
    col = db.get(Collection, collection_id)
    # 404 (not 403) for unreadable collections so existence isn't leaked
    if col is None or not auth.can_read(db, user, col):
        raise HTTPException(404, "Collection not found")
    if write and not auth.can_write(user, col):
        raise HTTPException(403, "Only the collection owner or an admin can modify it")
    return col


def _document(db: Session, user: User, document_id: int, write: bool = False) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _collection(db, user, doc.collection_id, write=write)
    return doc


def _doc_out(d: Document) -> dict:
    return {"id": d.id, "collection_id": d.collection_id, "filename": d.filename, "status": d.status,
            "error": d.error, "pages": d.n_pages, "chunks": d.n_chunks, "ocr_used": d.used_ocr,
            "flagged_chunks": d.flagged_chunks, "embedding_model": d.embedding_model,
            "uploaded_at": d.uploaded_at.isoformat()}


# ----------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/users", status_code=201)
def create_user(body: schemas.UserCreate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(409, "Username already exists")
    new, key = auth.create_user(db, body.username, body.role)
    return {"id": new.id, "username": new.username, "role": new.role, "api_key": key}


@app.get("/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "username": user.username, "role": user.role}


@app.get("/collections")
def list_collections(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [{"id": c.id, "name": c.name, "owner_id": c.owner_id, "documents": len(c.documents)}
            for c in auth.readable_collections(db, user)]


@app.post("/collections", status_code=201)
def create_collection(body: schemas.CollectionCreate, user: User = Depends(current_user),
                      db: Session = Depends(get_db)):
    col = Collection(name=body.name, owner_id=user.id)
    db.add(col)
    db.flush()
    return {"id": col.id, "name": col.name, "owner_id": col.owner_id, "documents": 0}


@app.post("/collections/{collection_id}/share")
def share_collection(collection_id: int, body: schemas.ShareRequest, user: User = Depends(current_user),
                     db: Session = Depends(get_db)):
    col = _collection(db, user, collection_id, write=True)
    target = db.scalar(select(User).where(User.username == body.username))
    if target is None:
        raise HTTPException(404, "User not found")
    if db.get(CollectionAccess, (col.id, target.id)) is None:
        db.add(CollectionAccess(collection_id=col.id, user_id=target.id))
    return {"shared_with": target.username}


@app.delete("/collections/{collection_id}", status_code=204)
def delete_collection(collection_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    col = _collection(db, user, collection_id, write=True)
    for doc in list(col.documents):
        delete_document(db, doc)
    for conv in db.scalars(select(Conversation).where(Conversation.collection_id == col.id)):
        db.delete(conv)
    for acc in db.scalars(select(CollectionAccess).where(CollectionAccess.collection_id == col.id)):
        db.delete(acc)
    db.delete(col)
    invalidate(collection_id)


@app.post("/collections/{collection_id}/documents", status_code=201)
async def upload_documents(collection_id: int, files: list[UploadFile] = File(...),
                           user: User = Depends(current_user), db: Session = Depends(get_db)):
    col = _collection(db, user, collection_id, write=True)
    results = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename or "").suffix) as tmp:
            tmp.write(await f.read())
            tmp_path = Path(tmp.name)
        try:
            doc, created = ingest_file(db, col, tmp_path, f.filename or "upload", user.id)
            results.append({**_doc_out(doc), "duplicate": not created})
        except ValueError as exc:
            results.append({"filename": f.filename, "status": "rejected", "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)
    return results


@app.get("/collections/{collection_id}/documents")
def list_documents(collection_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    col = _collection(db, user, collection_id)
    return [_doc_out(d) for d in col.documents]


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    delete_document(db, _document(db, user, document_id, write=True))


@app.post("/documents/{document_id}/reindex")
def reindex_document(document_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    doc = _document(db, user, document_id, write=True)
    return _doc_out(index_document(db, doc, user.id))


@app.get("/documents/{document_id}/preview")
def preview_document(document_id: int, page: int = 1, user: User = Depends(current_user),
                     db: Session = Depends(get_db)):
    doc = _document(db, user, document_id)
    chunks = db.scalars(select(Chunk).where(Chunk.document_id == doc.id, Chunk.page == page)
                        .order_by(Chunk.chunk_index)).all()
    if not chunks:
        raise HTTPException(404, f"No content on page {page}")
    return {"document_id": doc.id, "filename": doc.filename, "page": page, "total_pages": doc.n_pages,
            "heading": chunks[0].heading, "text": "\n\n".join(c.text for c in chunks)}


@app.post("/collections/{collection_id}/search")
def search_collection(collection_id: int, body: schemas.SearchRequest, user: User = Depends(current_user),
                      db: Session = Depends(get_db)):
    _collection(db, user, collection_id)
    hits = search(db, collection_id, body.query, top_k=body.top_k, mode=body.mode)
    return [{"chunk_id": h.chunk_id, "document_id": h.document_id, "filename": h.filename, "page": h.page,
             "heading": h.heading, "score": round(h.score, 4), "dense_rank": h.dense_rank,
             "bm25_rank": h.bm25_rank, "rerank_score": h.rerank_score, "text": h.text} for h in hits]


@app.post("/chat")
def chat(body: schemas.ChatRequest, user: User = Depends(current_user), db: Session = Depends(get_db)):
    _collection(db, user, body.collection_id)
    try:
        r = rag.chat(db, user.id, body.collection_id, body.question, body.conversation_id,
                     body.top_k, body.mode)
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    return {"conversation_id": r.conversation_id, "answer": r.answer, "citations": r.citations,
            "sources": r.sources, "usage": r.usage, "injection_redactions": r.injection_redactions}


@app.get("/conversations")
def list_conversations(collection_id: Optional[int] = None, user: User = Depends(current_user),
                       db: Session = Depends(get_db)):
    q = select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.id.desc())
    if collection_id:
        q = q.where(Conversation.collection_id == collection_id)
    return [{"id": c.id, "title": c.title, "collection_id": c.collection_id,
             "created_at": c.created_at.isoformat()} for c in db.scalars(q)]


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    return {"id": conv.id, "title": conv.title, "collection_id": conv.collection_id,
            "messages": [{"role": m.role, "content": m.content, "citations": m.citations or []}
                         for m in conv.messages]}


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)


@app.get("/usage")
def usage(scope: str = "me", user: User = Depends(current_user), db: Session = Depends(get_db)):
    if scope == "all":
        if user.role != "admin":
            raise HTTPException(403, "Admin only")
        return usage_summary(db)
    return usage_summary(db, user.id)
