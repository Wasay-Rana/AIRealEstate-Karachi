from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.models.internal import Chunk, RerankedDoc, RetrievedDoc


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"chunk_{i:02d}",
            document_id="doc_test",
            text=f"This is sample chunk number {i} about transformers and attention mechanisms.",
            token_count=15,
            metadata={"source": "test.txt", "document_id": "doc_test"},
            embedding=[0.1 * i] * 1536,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_retrieved_docs() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(
            chunk_id="chunk_00",
            text="Transformers use self-attention to relate positions in a sequence.",
            score=0.92,
            source="attention.txt",
            retriever="pinecone",
        ),
        RetrievedDoc(
            chunk_id="chunk_01",
            text="Entity linking maps text mentions to knowledge graph entities.",
            score=0.85,
            source="knowledge_graphs.txt",
            retriever="bm25",
        ),
        RetrievedDoc(
            chunk_id="chunk_02",
            text="RAG combines retrieval with generation for grounded answers.",
            score=0.78,
            source="rag_survey.txt",
            retriever="lightrag",
        ),
    ]


@pytest.fixture
def sample_reranked_docs() -> list[RerankedDoc]:
    return [
        RerankedDoc(
            chunk_id="chunk_00",
            text="Transformers use self-attention to relate positions in a sequence.",
            original_score=0.92,
            rerank_score=8.5,
            source="attention.txt",
        ),
        RerankedDoc(
            chunk_id="chunk_01",
            text="Entity linking maps text mentions to knowledge graph entities.",
            original_score=0.85,
            rerank_score=7.2,
            source="knowledge_graphs.txt",
        ),
        RerankedDoc(
            chunk_id="chunk_02",
            text="RAG combines retrieval with generation for grounded answers.",
            original_score=0.78,
            rerank_score=6.1,
            source="rag_survey.txt",
        ),
    ]


@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_batch.return_value = [[0.1] * 1536] * 5
    embedder.embed_one.return_value = [0.1] * 1536
    return embedder


@pytest.fixture
def mock_pinecone_store():
    store = MagicMock()
    store.ping.return_value = True
    store.upsert_chunks = AsyncMock(return_value=5)
    store.query = AsyncMock(
        return_value=[
            {
                "id": "chunk_00",
                "score": 0.92,
                "metadata": {"text": "Transformers use self-attention.", "source": "attention.txt"},
            }
        ]
    )
    store.fetch_all_metadata = AsyncMock(
        return_value=[
            {"id": "chunk_00", "text": "Transformers use self-attention.", "source": "attention.txt"},
            {"id": "chunk_01", "text": "Entity linking in knowledge graphs.", "source": "kg.txt"},
        ]
    )
    return store


@pytest.fixture
def mock_lightrag_store():
    store = MagicMock()
    store.ping.return_value = True
    store.insert_documents = AsyncMock()
    store.query = AsyncMock(
        return_value="The attention mechanism is analogous to entity linking. [Source 1]"
    )
    store.get_graph.return_value = None
    return store


@pytest.fixture
def mock_reranker():
    reranker = AsyncMock()
    reranker._model = MagicMock()

    async def rerank_side_effect(query, docs, top_n=None):
        n = top_n or len(docs)
        return [
            RerankedDoc(
                chunk_id=doc.chunk_id,
                text=doc.text,
                original_score=doc.score,
                rerank_score=doc.score * 10,
                source=doc.source,
            )
            for doc in docs[:n]
        ]

    reranker.rerank.side_effect = rerank_side_effect
    return reranker
