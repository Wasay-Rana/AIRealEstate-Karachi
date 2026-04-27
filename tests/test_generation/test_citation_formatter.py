from __future__ import annotations

import pytest

from app.generation.citation_formatter import extract_citations
from app.models.internal import RerankedDoc


def make_doc(chunk_id: str, source: str) -> RerankedDoc:
    return RerankedDoc(
        chunk_id=chunk_id,
        text="Sample text for citation testing.",
        original_score=0.9,
        rerank_score=8.0,
        source=source,
    )


def test_extracts_source_markers():
    docs = [
        make_doc("c1", "attention.txt"),
        make_doc("c2", "kg.txt"),
        make_doc("c3", "rag.txt"),
    ]
    answer = "Transformers use attention [Source 1]. KGs use entity linking [Source 2]."
    citations = extract_citations(answer, docs)

    sources = [c.source for c in citations]
    assert "attention.txt" in sources
    assert "kg.txt" in sources
    assert "rag.txt" not in sources  # Source 3 not mentioned


def test_no_markers_cites_all():
    docs = [make_doc(f"c{i}", f"doc{i}.txt") for i in range(3)]
    answer = "This answer has no source markers."
    citations = extract_citations(answer, docs)
    assert len(citations) == 3


def test_out_of_range_marker_ignored():
    docs = [make_doc("c1", "doc1.txt")]
    answer = "Some answer [Source 99] that references a missing source."
    citations = extract_citations(answer, docs)
    # Source 99 is out of range; fallback cites all docs
    assert len(citations) >= 1
