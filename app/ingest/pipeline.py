from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.ingest.chunker import SemanticChunker
from app.ingest.parsers import ParserFactory
from app.models.requests import IngestRequest

if TYPE_CHECKING:
    from app.ingest.embedder import OpenAIEmbedder
    from app.ingest.lightrag_store import LightRAGStore
    from app.ingest.pinecone_store import PineconeStore
    from app.ingest.status_tracker import IngestStatusTracker

logger = get_logger(__name__)


class IngestPipeline:
    def __init__(
        self,
        embedder: "OpenAIEmbedder",
        pinecone_store: "PineconeStore",
        lightrag_store: "LightRAGStore",
        status_tracker: "IngestStatusTracker",
    ) -> None:
        self._embedder = embedder
        self._pinecone = pinecone_store
        self._lightrag = lightrag_store
        self._tracker = status_tracker
        self._chunker = SemanticChunker()

    async def run(self, request: IngestRequest, job_id: str) -> None:
        self._tracker.update_job(job_id, "processing")

        try:
            document_id = str(uuid.uuid4())
            meta = {**request.metadata, "source": request.content[:120] if request.source_type == "url" else "uploaded"}

            parser = ParserFactory.get(request.source_type)
            text, parsed_meta = await parser.parse(request.content, meta)

            if not text.strip():
                raise ValueError("Parsed document is empty")

            chunks = self._chunker.chunk(text, document_id, parsed_meta)
            logger.info(f"[{job_id}] Created {len(chunks)} chunks from document")

            texts = [c.text for c in chunks]
            embeddings = await self._embedder.embed_batch(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            await asyncio.gather(
                self._pinecone.upsert_chunks(chunks, namespace=request.namespace),
                self._lightrag.insert_documents(texts),
            )

            self._tracker.update_job(job_id, "completed", chunks_created=len(chunks))
            logger.info(f"[{job_id}] Ingestion complete: {len(chunks)} chunks stored")

        except Exception as exc:
            logger.error(f"[{job_id}] Ingestion failed: {exc}", exc_info=True)
            self._tracker.update_job(job_id, "failed", error=str(exc))
            raise
