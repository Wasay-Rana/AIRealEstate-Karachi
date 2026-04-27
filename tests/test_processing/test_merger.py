from __future__ import annotations

import pytest

from app.models.internal import RetrievedDoc
from app.processing.merger import ResultMerger


def make_doc(chunk_id: str, score: float, retriever: str) -> RetrievedDoc:
    return RetrievedDoc(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=score,
        source="test.txt",
        retriever=retriever,
    )


def test_merge_deduplicates_same_chunk_id():
    docs = [
        make_doc("chunk_A", 0.9, "pinecone"),
        make_doc("chunk_A", 0.7, "bm25"),
        make_doc("chunk_B", 0.8, "pinecone"),
    ]
    merged = ResultMerger.merge(docs)
    ids = [d.chunk_id for d in merged]
    assert ids.count("chunk_A") == 1
    assert "chunk_B" in ids


def test_merge_rrf_orders_by_combined_rank():
    # chunk_A appears in both pinecone (rank 1) and bm25 (rank 1) → high RRF
    # chunk_B appears only in pinecone (rank 2) → lower RRF
    docs = [
        make_doc("chunk_A", 0.9, "pinecone"),
        make_doc("chunk_B", 0.8, "pinecone"),
        make_doc("chunk_A", 0.7, "bm25"),
    ]
    merged = ResultMerger.merge(docs)
    assert merged[0].chunk_id == "chunk_A"


def test_merge_empty_returns_empty():
    assert ResultMerger.merge([]) == []


def test_merge_single_retriever_preserves_order():
    docs = [
        make_doc("chunk_A", 0.9, "pinecone"),
        make_doc("chunk_B", 0.7, "pinecone"),
        make_doc("chunk_C", 0.5, "pinecone"),
    ]
    merged = ResultMerger.merge(docs)
    assert [d.chunk_id for d in merged] == ["chunk_A", "chunk_B", "chunk_C"]
