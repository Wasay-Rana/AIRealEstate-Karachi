from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ingest.embedder import OpenAIEmbedder
    from app.ingest.pinecone_store import PineconeStore
    from app.ingest.lightrag_store import LightRAGStore
    from app.ingest.status_tracker import IngestStatusTracker
    from app.processing.reranker import CrossEncoderReranker

# Module-level singletons populated by lifespan
_embedder: "OpenAIEmbedder | None" = None
_pinecone_store: "PineconeStore | None" = None
_lightrag_store: "LightRAGStore | None" = None
_status_tracker: "IngestStatusTracker | None" = None
_reranker: "CrossEncoderReranker | None" = None


def set_embedder(obj: "OpenAIEmbedder") -> None:
    global _embedder
    _embedder = obj


def set_pinecone_store(obj: "PineconeStore") -> None:
    global _pinecone_store
    _pinecone_store = obj


def set_lightrag_store(obj: "LightRAGStore") -> None:
    global _lightrag_store
    _lightrag_store = obj


def set_status_tracker(obj: "IngestStatusTracker") -> None:
    global _status_tracker
    _status_tracker = obj


def set_reranker(obj: "CrossEncoderReranker") -> None:
    global _reranker
    _reranker = obj


def get_embedder() -> "OpenAIEmbedder":
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    return _embedder


def get_pinecone_store() -> "PineconeStore":
    if _pinecone_store is None:
        raise RuntimeError("PineconeStore not initialized")
    return _pinecone_store


def get_lightrag_store() -> "LightRAGStore":
    if _lightrag_store is None:
        raise RuntimeError("LightRAGStore not initialized")
    return _lightrag_store


def get_status_tracker() -> "IngestStatusTracker":
    if _status_tracker is None:
        raise RuntimeError("StatusTracker not initialized")
    return _status_tracker


def get_reranker() -> "CrossEncoderReranker":
    if _reranker is None:
        raise RuntimeError("Reranker not initialized")
    return _reranker
