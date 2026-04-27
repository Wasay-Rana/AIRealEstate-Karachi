from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_mocks(mock_pinecone_store, mock_lightrag_store, mock_reranker):
    from app import core
    from app.core import dependencies as deps

    deps.set_pinecone_store(mock_pinecone_store)
    deps.set_lightrag_store(mock_lightrag_store)
    deps.set_reranker(mock_reranker)

    from app.api.health_router import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    return TestClient(app)


def test_health_returns_200(client_with_mocks):
    response = client_with_mocks.get("/api/v1/health")
    assert response.status_code == 200


def test_health_response_schema(client_with_mocks):
    data = client_with_mocks.get("/api/v1/health").json()
    assert "status" in data
    assert "components" in data
    assert "version" in data
    assert "uptime_seconds" in data
