from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str  # sha256(text)[:16]
    document_id: str
    text: str
    token_count: int
    metadata: dict[str, Any]
    embedding: Optional[list[float]] = None


class RetrievedDoc(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    retriever: str  # "pinecone" | "bm25" | "lightrag"
    metadata: dict[str, Any] = {}


class RerankedDoc(BaseModel):
    chunk_id: str
    text: str
    original_score: float
    rerank_score: float
    source: str
    metadata: dict[str, Any] = {}
