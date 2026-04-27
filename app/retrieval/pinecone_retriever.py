from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import RetrievedDoc

if TYPE_CHECKING:
    from app.ingest.pinecone_store import PineconeStore

logger = get_logger(__name__)


class PineconeRetriever:
    def __init__(self, store: "PineconeStore") -> None:
        self._store = store

    async def retrieve(
        self,
        embedding: list[float],
        top_k: int | None = None,
        namespace: str = "default",
    ) -> list[RetrievedDoc]:
        settings = get_settings()
        k = top_k or settings.pinecone_top_k

        matches = await self._store.query(embedding, top_k=k, namespace=namespace)

        docs: list[RetrievedDoc] = []
        for match in matches:
            meta = match.get("metadata") or {}
            docs.append(
                RetrievedDoc(
                    chunk_id=match["id"],
                    text=meta.get("text", ""),
                    score=float(match.get("score", 0.0)),
                    source=meta.get("source", "unknown"),
                    retriever="pinecone",
                    metadata=meta,
                )
            )

        logger.debug(f"Pinecone returned {len(docs)} docs")
        return docs
