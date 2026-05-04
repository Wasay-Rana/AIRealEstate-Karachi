from __future__ import annotations

import asyncio

import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_BATCH_SIZE = 100


class OpenAIEmbedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            embeddings = await self._embed_batch_raw(batch)
            results.extend(embeddings)
            if i + _BATCH_SIZE < len(texts):
                await asyncio.sleep(0.1)  # gentle rate-limit back-off

        logger.debug(f"Embedded {len(texts)} texts → {len(results)} vectors")
        return results

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    def as_lightrag_func(self):
        """Return an async embedding function compatible with LightRAG's interface."""
        from lightrag.utils import EmbeddingFunc

        settings = get_settings()
        embedder = self

        async def _func(texts: list[str]) -> np.ndarray:
            embeddings = await embedder.embed_batch(texts)
            return np.array(embeddings, dtype=np.float32)

        return EmbeddingFunc(
            embedding_dim=settings.embedding_dim,
            max_token_size=8192,
            func=_func,
        )
