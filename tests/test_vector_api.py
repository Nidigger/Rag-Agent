"""Unit tests for vector health check API endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


class TestVectorHealthEndpoint:
    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    @patch("app.api.v1.vector._get_vector_store")
    async def test_vector_health_healthy(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.get("/api/v1/vector/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "healthy"
        assert "provider" in data["data"]
        assert "collection" in data["data"]

    @patch("app.api.v1.vector._get_vector_store")
    async def test_vector_health_unhealthy(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = False
        mock_get_store.return_value = mock_store

        response = await client.get("/api/v1/vector/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "unhealthy"

    @patch("app.api.v1.vector._get_vector_store")
    async def test_vector_health_exception(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.side_effect = RuntimeError("Connection refused")
        mock_get_store.return_value = mock_store

        response = await client.get("/api/v1/vector/health")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "unhealthy"
