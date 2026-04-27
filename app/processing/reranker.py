from __future__ import annotations

import asyncio
from functools import partial

from sentence_transformers.cross_encoder import CrossEncoder

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import RerankedDoc, RetrievedDoc

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self) -> None:
        settings = get_settings()
        logger.info(f"Loading CrossEncoder: {settings.reranker_model}")
        self._model = CrossEncoder(settings.reranker_model)
        logger.info("CrossEncoder loaded")

    def _predict_sync(self, pairs: list[list[str]]) -> list[float]:
        return self._model.predict(pairs).tolist()

    async def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_n: int | None = None,
    ) -> list[RerankedDoc]:
        if not docs:
            return []

        settings = get_settings()
        n = top_n or settings.reranker_top_n

        pairs = [[query, doc.text] for doc in docs]

        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(
            None, partial(self._predict_sync, pairs)
        )

        reranked = [
            RerankedDoc(
                chunk_id=doc.chunk_id,
                text=doc.text,
                original_score=doc.score,
                rerank_score=float(score),
                source=doc.source,
                metadata=doc.metadata,
            )
            for doc, score in zip(docs, scores)
        ]

        reranked.sort(key=lambda d: d.rerank_score, reverse=True)
        top = reranked[:n]
        logger.debug(f"Reranker: {len(docs)} → {len(top)} docs")
        return top
