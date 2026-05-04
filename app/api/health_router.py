from __future__ import annotations

import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.dependencies import get_lightrag_store, get_pinecone_store, get_reranker
from app.core.logging import get_logger
from app.models.responses import HealthResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])

_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    components: dict[str, str] = {}
    overall = "healthy"

    try:
        ok = get_pinecone_store().ping()
        components["pinecone"] = "healthy" if ok else "unhealthy"
    except Exception as exc:
        components["pinecone"] = f"unhealthy: {exc}"
        overall = "degraded"

    try:
        ok = get_lightrag_store().ping()
        components["lightrag"] = "healthy" if ok else "unhealthy"
    except Exception as exc:
        components["lightrag"] = f"unhealthy: {exc}"
        overall = "degraded"

    try:
        reranker = get_reranker()
        components["reranker"] = "healthy" if reranker._model is not None else "unhealthy"
    except Exception as exc:
        components["reranker"] = f"unhealthy: {exc}"
        overall = "degraded"

    if any("unhealthy" in v for v in components.values()):
        overall = "unhealthy" if all("unhealthy" in v for v in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        components=components,
        version=settings.version,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )
