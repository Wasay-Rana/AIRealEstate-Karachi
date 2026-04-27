from __future__ import annotations

import re

from app.models.internal import RerankedDoc
from app.models.responses import Citation

_SOURCE_RE = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
_SNIPPET_LEN = 200


def extract_citations(answer: str, docs: list[RerankedDoc]) -> list[Citation]:
    """Parse [Source N] markers in the answer and map to docs."""
    used_indices: set[int] = set()
    for match in _SOURCE_RE.finditer(answer):
        idx = int(match.group(1)) - 1  # 1-based → 0-based
        if 0 <= idx < len(docs):
            used_indices.add(idx)

    # If no markers found, cite all docs
    if not used_indices:
        used_indices = set(range(len(docs)))

    citations: list[Citation] = []
    for idx in sorted(used_indices):
        doc = docs[idx]
        citations.append(
            Citation(
                source=doc.source,
                chunk_id=doc.chunk_id,
                score=round(doc.rerank_score, 4),
                text_snippet=doc.text[:_SNIPPET_LEN].strip(),
            )
        )

    return citations
