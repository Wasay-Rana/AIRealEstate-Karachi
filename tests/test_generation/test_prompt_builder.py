from __future__ import annotations

import pytest

from app.generation.prompt_builder import build_messages
from app.models.internal import RerankedDoc


def make_doc(chunk_id: str, text: str, source: str) -> RerankedDoc:
    return RerankedDoc(
        chunk_id=chunk_id,
        text=text,
        original_score=0.9,
        rerank_score=8.0,
        source=source,
    )


def test_build_messages_returns_correct_structure():
    docs = [make_doc("c1", "Transformers use self-attention.", "attention.txt")]
    system_blocks, messages = build_messages("What is self-attention?", docs)

    assert isinstance(system_blocks, list)
    assert len(system_blocks) >= 2
    assert messages[0]["role"] == "user"


def test_system_has_cache_control():
    docs = [make_doc("c1", "Some text.", "source.txt")]
    system_blocks, _ = build_messages("Test query", docs)

    # First block (static instructions) must have cache_control
    first_block = system_blocks[0]
    assert "cache_control" in first_block
    assert first_block["cache_control"]["type"] == "ephemeral"


def test_sources_numbered_correctly():
    docs = [
        make_doc("c1", "Text about RAG.", "rag.txt"),
        make_doc("c2", "Text about KGs.", "kg.txt"),
    ]
    system_blocks, _ = build_messages("Query", docs)
    context_block = system_blocks[1]["text"]

    assert "[Source 1]" in context_block
    assert "[Source 2]" in context_block
    assert "rag.txt" in context_block
    assert "kg.txt" in context_block


def test_query_in_user_message():
    docs = [make_doc("c1", "Some text.", "src.txt")]
    _, messages = build_messages("What is HNSW?", docs)
    assert "What is HNSW?" in messages[0]["content"]
