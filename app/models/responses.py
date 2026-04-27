from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    text_snippet: str


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    chunks_created: Optional[int] = None
    message: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    mode_used: str
    query_rewritten: Optional[str] = None
    latency_ms: float
    retrieval_breakdown: dict[str, int]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "entity"
    description: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str = ""
    weight: float = 1.0


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_nodes: int
    total_edges: int


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    components: dict[str, str]
    version: str
    uptime_seconds: float
