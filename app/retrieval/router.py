from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.dependencies import (
    get_embedder,
    get_lightrag_store,
    get_pinecone_store,
    get_reranker,
)
from app.core.logging import get_logger
from app.generation.generator import AnswerGenerator
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse
from app.processing.compressor import ContextCompressor
from app.processing.merger import ResultMerger
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.orchestrator import RetrievalOrchestrator

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])

# BM25 is module-level so the same index is shared across requests
_bm25 = BM25Retriever()


def get_bm25() -> BM25Retriever:
    return _bm25


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    orchestrator = RetrievalOrchestrator(
        embedder=get_embedder(),
        pinecone_store=get_pinecone_store(),
        lightrag_store=get_lightrag_store(),
        bm25_retriever=get_bm25(),
    )

    # Retrieve
    raw_docs, effective_query, resolved_mode = await orchestrator.retrieve(request)

    # Merge + dedup
    merged = ResultMerger.merge(raw_docs)

    # Rerank
    reranker = get_reranker()
    top_n = min(request.top_k, len(merged))
    reranked = await reranker.rerank(effective_query, merged, top_n=top_n)

    # Compress context
    compressed = ContextCompressor.compress(reranked)

    # Generate answer
    generator = AnswerGenerator()
    answer, citations = await generator.generate(effective_query, compressed)

    latency = (time.perf_counter() - start) * 1000

    retrieval_breakdown = {
        "pinecone": sum(1 for d in raw_docs if d.retriever == "pinecone"),
        "bm25": sum(1 for d in raw_docs if d.retriever == "bm25"),
        "lightrag": sum(1 for d in raw_docs if d.retriever == "lightrag"),
    }

    return QueryResponse(
        answer=answer,
        citations=citations,
        mode_used=resolved_mode,
        query_rewritten=effective_query if effective_query != request.query else None,
        latency_ms=round(latency, 1),
        retrieval_breakdown=retrieval_breakdown,
    )
