from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import RetrievedDoc
from app.models.requests import QueryRequest
from app.retrieval.complexity_detector import detect_mode
from app.retrieval.query_rewriter import QueryRewriter

if TYPE_CHECKING:
    from app.ingest.embedder import OpenAIEmbedder
    from app.ingest.lightrag_store import LightRAGStore
    from app.ingest.pinecone_store import PineconeStore
    from app.retrieval.bm25_retriever import BM25Retriever

logger = get_logger(__name__)


class RetrievalOrchestrator:
    def __init__(
        self,
        embedder: "OpenAIEmbedder",
        pinecone_store: "PineconeStore",
        lightrag_store: "LightRAGStore",
        bm25_retriever: "BM25Retriever",
    ) -> None:
        from app.retrieval.lightrag_retriever import LightRAGRetriever
        from app.retrieval.pinecone_retriever import PineconeRetriever

        self._embedder = embedder
        self._pinecone_retriever = PineconeRetriever(pinecone_store)
        self._lightrag_retriever = LightRAGRetriever(lightrag_store)
        self._bm25 = bm25_retriever
        self._rewriter = QueryRewriter()

    async def retrieve(
        self,
        request: QueryRequest,
    ) -> tuple[list[RetrievedDoc], str, str]:
        """Returns (docs, effective_query, resolved_mode)."""
        settings = get_settings()

        # 1. Query rewriting
        effective_query = request.query
        if request.rewrite_query:
            effective_query = await self._rewriter.rewrite(request.query)

        # 2. Mode resolution
        if request.mode == "auto":
            resolved_mode = detect_mode(effective_query)
        else:
            resolved_mode = request.mode

        logger.info(f"Retrieval mode={resolved_mode} query={effective_query!r:.80}")

        # 3. Embed query for dense retrieval
        embedding = await self._embedder.embed_one(effective_query)

        # 4. Fan-out based on mode
        all_docs: list[RetrievedDoc] = []

        if resolved_mode == "fast":
            docs = await self._pinecone_retriever.retrieve(
                embedding, top_k=request.top_k * 2, namespace=request.namespace
            )
            all_docs.extend(docs)

        elif resolved_mode == "balanced":
            pinecone_docs, bm25_docs = await asyncio.gather(
                self._pinecone_retriever.retrieve(
                    embedding, top_k=request.top_k * 2, namespace=request.namespace
                ),
                asyncio.get_event_loop().run_in_executor(
                    None, self._bm25.retrieve, effective_query, request.top_k
                ),
            )
            all_docs.extend(pinecone_docs)
            all_docs.extend(bm25_docs)

        elif resolved_mode == "deep":
            pinecone_docs, lg_docs = await asyncio.gather(
                self._pinecone_retriever.retrieve(
                    embedding, top_k=request.top_k, namespace=request.namespace
                ),
                self._lightrag_retriever.retrieve(
                    effective_query,
                    mode=request.lightrag_mode,
                    top_k=request.top_k,
                ),
            )
            all_docs.extend(pinecone_docs)
            all_docs.extend(lg_docs)

        logger.info(
            f"Retrieved {len(all_docs)} docs total "
            f"(pinecone={sum(1 for d in all_docs if d.retriever=='pinecone')}, "
            f"bm25={sum(1 for d in all_docs if d.retriever=='bm25')}, "
            f"lightrag={sum(1 for d in all_docs if d.retriever=='lightrag')})"
        )

        return all_docs, effective_query, resolved_mode
