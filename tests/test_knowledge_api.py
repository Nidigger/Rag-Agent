"""Unit tests for knowledge base ingest/delete API endpoints."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.config import settings
from app.main import app

_INTERNAL_TOKEN = "test-dev-token-12345"


@pytest.fixture(autouse=True)
def setup_internal_token(monkeypatch):
    monkeypatch.setattr(
        settings.security,
        "internal_token",
        SecretStr(_INTERNAL_TOKEN),
    )


class TestKnowledgeIngestEndpoint:
    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    def _auth_headers(self):
        return {"X-Internal-Token": _INTERNAL_TOKEN}

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_with_valid_token_succeeds(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("API test document content for ingest endpoint." * 15)
            tmp.close()

            mock_store.upsert_chunks = MagicMock()
            response = await client.post(
                "/api/v1/internal/knowledge/documents/doc_api_test/ingest",
                json={
                    "document_id": "doc_api_test",
                    "file_path": tmp.name,
                    "knowledge_base_id": "kb_test",
                },
                headers=self._auth_headers(),
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert data["data"]["document_id"] == "doc_api_test"
            assert data["data"]["status"] == "success"
        finally:
            os.unlink(tmp.name)

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_without_token_rejected(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_no_token/ingest",
            json={
                "document_id": "doc_no_token",
                "file_path": "/tmp/somefile.txt",
            },
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == "UNAUTHORIZED"

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_with_wrong_token_rejected(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_bad_token/ingest",
            json={
                "document_id": "doc_bad_token",
                "file_path": "/tmp/somefile.txt",
            },
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert response.status_code == 401
        assert response.json()["error_code"] == "UNAUTHORIZED"

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_file_not_found(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_nonexistent/ingest",
            json={
                "document_id": "doc_nonexistent",
                "file_path": "data/nonexistent_file.txt",
            },
            headers=self._auth_headers(),
        )
        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "DOCUMENT_NOT_FOUND"

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_vector_store_unavailable(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = False
        mock_get_store.return_value = mock_store

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("content")
            tmp.close()

            response = await client.post(
                "/api/v1/internal/knowledge/documents/doc_unavailable/ingest",
                json={
                    "document_id": "doc_unavailable",
                    "file_path": tmp.name,
                },
                headers=self._auth_headers(),
            )
            assert response.status_code == 503
            data = response.json()
            assert data["error_code"] == "VECTOR_STORE_UNAVAILABLE"
        finally:
            os.unlink(tmp.name)

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_path_traversal_rejected(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_traversal/ingest",
            json={
                "document_id": "doc_traversal",
                "file_path": "../../etc/passwd",
            },
            headers=self._auth_headers(),
        )
        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "UNAUTHORIZED"


class TestKnowledgeDeleteEndpoint:
    @pytest.fixture
    async def client(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    def _auth_headers(self):
        return {"X-Internal-Token": _INTERNAL_TOKEN}

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_delete_document_logical(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_store.disable_document = MagicMock()
        mock_get_store.return_value = mock_store

        response = await client.request(
            "DELETE",
            "/api/v1/internal/knowledge/documents/doc_delete/vectors",
            headers=self._auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["document_id"] == "doc_delete"
        assert data["data"]["status"] == "disabled"

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_delete_document_physical(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_store.delete_document = MagicMock()
        mock_get_store.return_value = mock_store

        response = await client.request(
            "DELETE",
            "/api/v1/internal/knowledge/documents/doc_purge/vectors",
            json={"document_id": "doc_purge", "physical": True},
            headers=self._auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "deleted"

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_delete_without_token_rejected(self, mock_get_store, client):
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.request(
            "DELETE",
            "/api/v1/internal/knowledge/documents/doc_noauth/vectors",
        )
        assert response.status_code == 401
