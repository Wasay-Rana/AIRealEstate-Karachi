from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import RetrievedDoc

if TYPE_CHECKING:
    from app.ingest.pinecone_store import PineconeStore

logger = get_logger(__name__)

_CACHE_PATH = Path(".bm25_corpus.pkl")


class BM25Retriever:
    """
    In-memory BM25 index built from Pinecone metadata at startup.
    Suitable for corpora up to ~50K chunks. For larger corpora,
    replace with Elasticsearch.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []

    async def build_index(self, store: "PineconeStore", namespace: str = "default") -> None:
        settings = get_settings()

        if _CACHE_PATH.exists():
            logger.info("Loading BM25 corpus from cache")
            with _CACHE_PATH.open("rb") as f:
                self._corpus = pickle.load(f)
        else:
            logger.info("Building BM25 index from Pinecone metadata…")
            self._corpus = await store.fetch_all_metadata(namespace=namespace)
            with _CACHE_PATH.open("wb") as f:
                pickle.dump(self._corpus, f)

        if not self._corpus:
            logger.warning("BM25 corpus is empty — keyword search disabled")
            return

        tokenized = [doc.get("text", "").lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(self._corpus)} documents")

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        settings = get_settings()
        k = top_k or settings.bm25_top_k

        if self._bm25 is None or not self._corpus:
            logger.debug("BM25 index empty — returning no results")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        docs: list[RetrievedDoc] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            doc_meta = self._corpus[idx]
            docs.append(
                RetrievedDoc(
                    chunk_id=doc_meta.get("id", f"bm25_{idx}"),
                    text=doc_meta.get("text", ""),
                    score=float(scores[idx]),
                    source=doc_meta.get("source", "unknown"),
                    retriever="bm25",
                    metadata=doc_meta,
                )
            )

        logger.debug(f"BM25 returned {len(docs)} docs")
        return docs
