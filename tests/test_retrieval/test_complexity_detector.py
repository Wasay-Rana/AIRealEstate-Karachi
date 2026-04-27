from __future__ import annotations

import pytest

from app.retrieval.complexity_detector import detect_mode


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is RAG?", "fast"),
        ("Define attention", "fast"),
        (
            "How does the attention mechanism in Transformers relate to entity linking in KGs?",
            "deep",
        ),
        ("Compare Pinecone and Qdrant and explain why HNSW is better", "deep"),
        (
            "What is the difference between naive RAG and modular RAG?",
            "deep",
        ),
        (
            "Explain how vector databases work and what makes Pinecone scalable",
            "balanced",
        ),
    ],
)
def test_detect_mode(query: str, expected: str):
    result = detect_mode(query)
    assert result == expected, f"Expected {expected!r} for {query!r}, got {result!r}"
