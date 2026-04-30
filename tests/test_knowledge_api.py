"""Unit tests for knowledge base ingest/delete API endpoints."""

import os
import tempfile
from pathlib import Path
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

    @patch("app.api.v1.knowledge.get_storage")
    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_with_minio_storage_succeeds(
        self, mock_get_store, mock_get_storage, client
    ):
        """MinIO storage mode downloads file and ingests successfully."""
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        # Create a temp file to simulate MinIO download
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write("MinIO test document content for ingest." * 15)
        tmp.close()

        from app.storage.base import LocalFileRef

        local_ref = LocalFileRef(
            path=tmp.name,
            filename="manual.txt",
            should_cleanup=True,
        )

        mock_storage = MagicMock()
        mock_storage.download_to_temp.return_value = local_ref
        mock_get_storage.return_value = mock_storage

        try:
            response = await client.post(
                "/api/v1/internal/knowledge/documents/doc_minio/ingest",
                json={
                    "document_id": "doc_minio",
                    "storage": {
                        "provider": "minio",
                        "bucket": "rag-agent",
                        "object_key": "original/default/kb_default/doc_minio/v1/manual.txt",
                        "file_name": "manual.txt",
                    },
                    "knowledge_base_id": "kb_default",
                },
                headers=self._auth_headers(),
            )
            assert response.status_code == 200
            data = response.json()
            assert data["code"] == 0
            assert data["data"]["document_id"] == "doc_minio"
            assert data["data"]["status"] == "success"

            mock_storage.download_to_temp.assert_called_once_with(
                bucket="rag-agent",
                object_key="original/default/kb_default/doc_minio/v1/manual.txt",
            )
        finally:
            # File may have been cleaned up by the endpoint's finally block
            Path(tmp.name).unlink(missing_ok=True)

    @patch("app.api.v1.knowledge.get_storage")
    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_minio_temp_file_cleaned_up(
        self, mock_get_store, mock_get_storage, client
    ):
        """MinIO temp file is cleaned up even when ingest fails."""
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write("content to be cleaned")
        tmp.close()
        temp_path = tmp.name

        from app.storage.base import LocalFileRef

        local_ref = LocalFileRef(
            path=temp_path,
            filename="cleanme.txt",
            should_cleanup=True,
        )

        mock_storage = MagicMock()
        mock_storage.download_to_temp.return_value = local_ref
        mock_get_storage.return_value = mock_storage

        # Make ingest fail to verify cleanup still happens
        mock_store.upsert_chunks.side_effect = Exception("Ingest explosion")

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_cleanup/ingest",
            json={
                "document_id": "doc_cleanup",
                "storage": {
                    "provider": "minio",
                    "bucket": "rag-agent",
                    "object_key": "original/default/kb/doc/v1/cleanme.txt",
                },
            },
            headers=self._auth_headers(),
        )
        assert response.status_code == 500

        # Verify temp file was cleaned up
        assert not Path(temp_path).exists()

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_both_file_path_and_storage_rejected(
        self, mock_get_store, client
    ):
        """Request with both file_path and storage should fail validation."""
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_both/ingest",
            json={
                "document_id": "doc_both",
                "file_path": "data/test.txt",
                "storage": {
                    "provider": "minio",
                    "object_key": "key",
                },
            },
            headers=self._auth_headers(),
        )
        assert response.status_code == 422

    @patch("app.api.v1.knowledge._get_vector_store")
    async def test_ingest_neither_file_path_nor_storage_rejected(
        self, mock_get_store, client
    ):
        """Request with neither file_path nor storage should fail validation."""
        mock_store = MagicMock()
        mock_store.health_check.return_value = True
        mock_get_store.return_value = mock_store

        response = await client.post(
            "/api/v1/internal/knowledge/documents/doc_neither/ingest",
            json={
                "document_id": "doc_neither",
            },
            headers=self._auth_headers(),
        )
        assert response.status_code == 422


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
