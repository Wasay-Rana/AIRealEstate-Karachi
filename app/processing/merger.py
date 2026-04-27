from __future__ import annotations

from app.models.internal import RetrievedDoc

_RRF_K = 60


class ResultMerger:
    @staticmethod
    def merge(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """
        Dedup by chunk_id then combine scores using Reciprocal Rank Fusion.
        RRF_score(d) = sum(1 / (k + rank_i)) across all retrievers that returned d.
        """
        if not docs:
            return []

        # Group by retriever to compute per-retriever ranks
        by_retriever: dict[str, list[RetrievedDoc]] = {}
        for doc in docs:
            by_retriever.setdefault(doc.retriever, []).append(doc)

        # Sort each retriever's list by score descending
        for lst in by_retriever.values():
            lst.sort(key=lambda d: d.score, reverse=True)

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        best_doc: dict[str, RetrievedDoc] = {}

        for lst in by_retriever.values():
            for rank, doc in enumerate(lst):
                rrf_scores[doc.chunk_id] = rrf_scores.get(doc.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                # Keep the version with highest original score
                if doc.chunk_id not in best_doc or doc.score > best_doc[doc.chunk_id].score:
                    best_doc[doc.chunk_id] = doc

        # Build deduplicated list sorted by RRF score
        merged = sorted(best_doc.values(), key=lambda d: rrf_scores[d.chunk_id], reverse=True)

        # Inject RRF score as the doc's score for downstream use
        for doc in merged:
            doc.score = rrf_scores[doc.chunk_id]

        return merged
