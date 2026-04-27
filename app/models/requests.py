from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_type: Literal["pdf", "text", "url"]
    content: str  # base64 for PDF, raw text, or URL string
    metadata: dict[str, Any] = {}
    namespace: str = "default"
    background: bool = True


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["auto", "fast", "balanced", "deep"] = "auto"
    lightrag_mode: Literal["local", "global", "hybrid", "mix"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    namespace: str = "default"
    stream: bool = False
    rewrite_query: bool = True


class GraphExploreRequest(BaseModel):
    entity: Optional[str] = None
    depth: int = Field(default=2, ge=1, le=5)
