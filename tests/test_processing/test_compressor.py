from __future__ import annotations

import pytest

from app.models.internal import RerankedDoc
from app.processing.compressor import ContextCompressor


def make_doc(chunk_id: str, text: str) -> RerankedDoc:
    return RerankedDoc(
        chunk_id=chunk_id,
        text=text,
        original_score=0.9,
        rerank_score=8.0,
        source="test.txt",
    )


def test_compress_within_budget_keeps_all():
    docs = [make_doc(f"c{i}", "word " * 50) for i in range(3)]
    compressed = ContextCompressor.compress(docs, max_tokens=1000)
    assert len(compressed) == 3


def test_compress_exceeds_budget_truncates():
    # Each doc is ~50 tokens; budget = 80 → fits 1 full + partial
    docs = [make_doc(f"c{i}", "word " * 50) for i in range(5)]
    compressed = ContextCompressor.compress(docs, max_tokens=80)
    assert len(compressed) <= 2


def test_compress_empty_returns_empty():
    assert ContextCompressor.compress([]) == []


def test_compress_single_large_doc_truncates():
    long_text = "This is a sentence. " * 500  # ~1000 tokens
    docs = [make_doc("big", long_text)]
    compressed = ContextCompressor.compress(docs, max_tokens=100)
    assert len(compressed) == 1
    from app.utils.text_utils import count_tokens
    assert count_tokens(compressed[0].text) <= 100
