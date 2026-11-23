from typing import Optional

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    role: str = Field(default="user", pattern="^(admin|user)$")


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class ShareRequest(BaseModel):
    username: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid_rerank", pattern="^(dense|bm25|hybrid|hybrid_rerank)$")


class ChatRequest(BaseModel):
    collection_id: int
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[int] = None
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid_rerank", pattern="^(dense|bm25|hybrid|hybrid_rerank)$")
