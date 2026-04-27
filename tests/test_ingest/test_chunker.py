from __future__ import annotations

import pytest

from app.ingest.chunker import SemanticChunker


@pytest.fixture
def chunker():
    return SemanticChunker(chunk_size=100, chunk_overlap=10)


def test_chunker_produces_chunks(chunker):
    text = "The Transformer architecture uses self-attention. " * 30
    chunks = chunker.chunk(text, "doc_1", {"source": "test.txt"})
    assert len(chunks) > 1


def test_chunker_chunk_ids_are_unique(chunker):
    text = "The Transformer architecture uses self-attention. " * 30
    chunks = chunker.chunk(text, "doc_1", {"source": "test.txt"})
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunker_metadata_propagated(chunker):
    text = "Some text about knowledge graphs and retrieval systems. " * 10
    chunks = chunker.chunk(text, "doc_42", {"source": "kg.txt", "author": "test"})
    for chunk in chunks:
        assert chunk.document_id == "doc_42"
        assert chunk.metadata.get("source") == "kg.txt"


def test_chunker_empty_text_returns_empty(chunker):
    chunks = chunker.chunk("", "doc_x", {})
    assert chunks == []


def test_chunker_token_count_is_positive(chunker):
    text = "Knowledge graphs link entities through relationships. " * 20
    chunks = chunker.chunk(text, "doc_1", {})
    for chunk in chunks:
        assert chunk.token_count > 0
