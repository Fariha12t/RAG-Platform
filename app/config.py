"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # storage
    database_url: str
    data_dir: Path
    # embeddings / retrieval
    embedding_backend: str      # hashing | sbert | openai
    embedding_model: str
    embedding_dim: int          # only used by the hashing backend
    vector_backend: str         # numpy | faiss
    reranker_backend: str       # lexical | cross-encoder | none
    reranker_model: str
    # generation
    llm_provider: str           # extractive | anthropic | openai
    anthropic_model: str
    openai_model: str
    max_output_tokens: int
    # ingestion
    chunk_size: int
    chunk_overlap: int
    ocr_enabled: bool
    ocr_min_chars: int
    max_upload_mb: int
    # retrieval defaults
    top_k: int
    candidate_k: int
    # auth bootstrap
    admin_username: str
    admin_api_key: str
    # cost tracking
    pricing_overrides: dict

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "./storage")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        database_url=os.getenv("DATABASE_URL", f"sqlite:///{data_dir / 'rag.db'}"),
        data_dir=data_dir,
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "hashing"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        vector_backend=os.getenv("VECTOR_BACKEND", "numpy"),
        reranker_backend=os.getenv("RERANKER_BACKEND", "lexical"),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        llm_provider=os.getenv("LLM_PROVIDER", "extractive"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "800")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        ocr_enabled=_bool("OCR_ENABLED", True),
        ocr_min_chars=int(os.getenv("OCR_MIN_CHARS", "25")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
        top_k=int(os.getenv("TOP_K", "5")),
        candidate_k=int(os.getenv("CANDIDATE_K", "25")),
        admin_username=os.getenv("ADMIN_USERNAME", "admin"),
        admin_api_key=os.getenv("ADMIN_API_KEY", "dev-admin-key"),
        pricing_overrides=json.loads(os.getenv("PRICING_JSON", "{}") or "{}"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings() -> None:
    """Clear cached settings (used by tests and the evaluation harness)."""
    get_settings.cache_clear()
