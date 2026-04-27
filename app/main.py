from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import dependencies as deps
from app.core.config import get_settings
from app.core.exceptions import (
    GenerationError,
    IngestError,
    RetrievalError,
    generation_error_handler,
    ingest_error_handler,
    retrieval_error_handler,
)
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info(f"Starting Graph-Enhanced RAG Agent v{settings.version} [{settings.environment}]")
    start = time.time()

    # 1. Embedder
    from app.ingest.embedder import OpenAIEmbedder
    embedder = OpenAIEmbedder()
    deps.set_embedder(embedder)

    # 2. Pinecone
    from app.ingest.pinecone_store import PineconeStore
    pinecone_store = PineconeStore()
    pinecone_store.initialize()
    deps.set_pinecone_store(pinecone_store)

    # 3. LightRAG (async init)
    from app.ingest.lightrag_store import LightRAGStore
    lightrag_store = LightRAGStore(embedder=embedder)
    await lightrag_store.initialize()
    deps.set_lightrag_store(lightrag_store)

    # 4. Status tracker
    from app.ingest.status_tracker import IngestStatusTracker
    tracker = IngestStatusTracker()
    deps.set_status_tracker(tracker)

    # 5. CrossEncoder reranker (loads model into memory)
    from app.processing.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    deps.set_reranker(reranker)

    # 6. Build BM25 index from Pinecone corpus
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.router import get_bm25
    bm25 = get_bm25()
    await bm25.build_index(pinecone_store)

    logger.info(f"All components initialized in {time.time() - start:.1f}s — ready")
    yield

    logger.info("Shutting down Graph-Enhanced RAG Agent")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Graph-Enhanced RAG Agent",
        description=(
            "Production-ready RAG system combining LightRAG knowledge graphs, "
            "Pinecone vector search, and Claude reasoning."
        ),
        version=settings.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(IngestError, ingest_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RetrievalError, retrieval_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(GenerationError, generation_error_handler)  # type: ignore[arg-type]

    # Routers
    from app.ingest.router import router as ingest_router
    from app.retrieval.router import router as query_router
    from app.api.graph_router import router as graph_router
    from app.api.health_router import router as health_router
    from app.api.property_router import router as property_router

    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(graph_router)
    app.include_router(health_router)
    app.include_router(property_router)

    return app


app = create_app()
