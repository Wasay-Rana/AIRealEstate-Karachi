from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import RetrievedDoc

if TYPE_CHECKING:
    from app.ingest.lightrag_store import LightRAGStore

logger = get_logger(__name__)


class LightRAGRetriever:
    def __init__(self, store: "LightRAGStore") -> None:
        self._store = store

    async def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> list[RetrievedDoc]:
        settings = get_settings()
        k = top_k or settings.lightrag_top_k

        answer_text = await self._store.query(query, mode=mode, top_k=k)

        if not answer_text or not answer_text.strip():
            return []

        # LightRAG returns a synthesized answer — we wrap it as a retrieved doc
        # so it flows through the same merge/rerank pipeline.
        chunk_id = "lg_" + hashlib.sha256(answer_text.encode()).hexdigest()[:12]

        doc = RetrievedDoc(
            chunk_id=chunk_id,
            text=answer_text,
            score=1.0,  # Graph result is highly relevant by construction
            source="lightrag_graph",
            retriever="lightrag",
            metadata={"mode": mode, "query": query},
        )

        logger.debug(f"LightRAG ({mode}) returned {len(answer_text)} chars")
        return [doc]
