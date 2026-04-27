from __future__ import annotations

import asyncio
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.internal import Chunk

logger = get_logger(__name__)
_UPSERT_BATCH = 100


class PineconeStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._dim = settings.embedding_dim
        self._index = None

    def initialize(self) -> None:
        """Create index if it doesn't exist and connect."""
        settings = get_settings()
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing:
            logger.info(f"Creating Pinecone index '{self._index_name}' dim={self._dim}")
            self._pc.create_index(
                name=self._index_name,
                dimension=self._dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
            )
        self._index = self._pc.Index(self._index_name)
        logger.info(f"Connected to Pinecone index '{self._index_name}'")

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError("PineconeStore not initialized — call initialize() first")
        return self._index

    async def upsert_chunks(self, chunks: list[Chunk], namespace: str = "default") -> int:
        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding — skipping")
                continue
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": chunk.embedding,
                    "metadata": {
                        "text": chunk.text[:1000],  # Pinecone metadata value limit
                        "source": chunk.metadata.get("source", "unknown"),
                        "document_id": chunk.document_id,
                        "token_count": chunk.token_count,
                        **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
                    },
                }
            )

        if not vectors:
            return 0

        # Batch upsert
        loop = asyncio.get_event_loop()
        total = 0
        for i in range(0, len(vectors), _UPSERT_BATCH):
            batch = vectors[i : i + _UPSERT_BATCH]
            await loop.run_in_executor(
                None, lambda b=batch: self.index.upsert(vectors=b, namespace=namespace)
            )
            total += len(batch)

        logger.info(f"Upserted {total} vectors to Pinecone namespace='{namespace}'")
        return total

    async def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        namespace: str = "default",
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        if filter:
            kwargs["filter"] = filter

        result = await loop.run_in_executor(None, lambda: self.index.query(**kwargs))
        return result.get("matches", [])

    async def fetch_all_metadata(self, namespace: str = "default", limit: int = 10000) -> list[dict[str, Any]]:
        """Fetch metadata for BM25 corpus building. Uses list + fetch."""
        loop = asyncio.get_event_loop()
        try:
            list_result = await loop.run_in_executor(
                None, lambda: self.index.list(namespace=namespace, limit=limit)
            )
            ids = list(list_result)
            if not ids:
                return []
            fetch_result = await loop.run_in_executor(
                None, lambda: self.index.fetch(ids=ids, namespace=namespace)
            )
            return [
                {"id": vid, **vec.metadata}
                for vid, vec in fetch_result.vectors.items()
                if vec.metadata
            ]
        except Exception as e:
            logger.warning(f"Could not fetch all metadata: {e}")
            return []

    def ping(self) -> bool:
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False
