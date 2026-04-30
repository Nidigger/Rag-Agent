"""Tests for app/storage/ — MinIO storage backend."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from app.config.schema import StorageSettings
from app.storage import get_storage
from app.storage.base import LocalFileRef, ObjectStorage
from app.storage.minio_storage import MinioStorage


class TestObjectStorageBase:
    def test_download_to_temp_raises_not_implemented(self):
        base = ObjectStorage()
        with pytest.raises(NotImplementedError):
            base.download_to_temp("bucket", "key")


class TestLocalFileRef:
    def test_defaults(self):
        ref = LocalFileRef(path="/tmp/test.txt", filename="test.txt")
        assert ref.path == "/tmp/test.txt"
        assert ref.filename == "test.txt"
        assert ref.content_type is None
        assert ref.should_cleanup is True

    def test_custom_values(self):
        ref = LocalFileRef(
            path="/tmp/test.pdf",
            filename="test.pdf",
            content_type="application/pdf",
            should_cleanup=False,
        )
        assert ref.content_type == "application/pdf"
        assert ref.should_cleanup is False


class TestMinioStorage:
    @patch("app.storage.minio_storage.settings")
    def test_download_to_temp_calls_fget_object(self, mock_settings):
        mock_settings.storage = StorageSettings(
            provider="minio",
            minio_endpoint="192.168.31.104:9000",
            minio_bucket="rag-agent",
            minio_access_key=SecretStr("testkey"),
            minio_secret_key=SecretStr("testsecret"),
            minio_secure=False,
        )

        with patch("app.storage.minio_storage.Minio") as MockMinio:
            mock_client = MagicMock()
            MockMinio.return_value = mock_client

            storage = MinioStorage()
            result = storage.download_to_temp(
                bucket="rag-agent",
                object_key="original/default/kb/doc/v1/manual.pdf",
            )

        assert isinstance(result, LocalFileRef)
        assert result.filename == "manual.pdf"
        assert result.path.endswith(".pdf")
        assert result.should_cleanup is True

        mock_client.fget_object.assert_called_once()
        call_args = mock_client.fget_object.call_args
        assert call_args[0][0] == "rag-agent"
        assert call_args[0][1] == "original/default/kb/doc/v1/manual.pdf"

        # Cleanup temp file
        Path(result.path).unlink(missing_ok=True)

    @patch("app.storage.minio_storage.settings")
    def test_download_to_temp_preserves_extension(self, mock_settings):
        mock_settings.storage = StorageSettings(
            provider="minio",
            minio_endpoint="192.168.31.104:9000",
            minio_bucket="rag-agent",
            minio_access_key=SecretStr("testkey"),
            minio_secret_key=SecretStr("testsecret"),
            minio_secure=False,
        )

        with patch("app.storage.minio_storage.Minio") as MockMinio:
            mock_client = MagicMock()
            MockMinio.return_value = mock_client

            storage = MinioStorage()
            result = storage.download_to_temp(
                bucket="rag-agent",
                object_key="original/default/kb/doc/v1/readme.txt",
            )

        assert result.path.endswith(".txt")
        assert result.filename == "readme.txt"

        Path(result.path).unlink(missing_ok=True)

    @patch("app.storage.minio_storage.settings")
    def test_download_failure_raises_runtime_error(self, mock_settings):
        from minio.error import S3Error

        mock_settings.storage = StorageSettings(
            provider="minio",
            minio_endpoint="192.168.31.104:9000",
            minio_bucket="rag-agent",
            minio_access_key=SecretStr("testkey"),
            minio_secret_key=SecretStr("testsecret"),
            minio_secure=False,
        )

        with patch("app.storage.minio_storage.Minio") as MockMinio:
            mock_client = MagicMock()
            mock_client.fget_object.side_effect = S3Error(
                "NoSuchKey",
                "The specified key does not exist.",
                "resource",
                "request_id",
                "host_id",
                "response",
            )
            MockMinio.return_value = mock_client

            storage = MinioStorage()
            with pytest.raises(RuntimeError, match="MinIO download failed"):
                storage.download_to_temp(
                    bucket="rag-agent",
                    object_key="nonexistent/key.pdf",
                )

    @patch("app.storage.minio_storage.settings")
    def test_endpoint_strips_http_prefix(self, mock_settings):
        mock_settings.storage = StorageSettings(
            provider="minio",
            minio_endpoint="http://192.168.31.104:9000",
            minio_bucket="rag-agent",
            minio_access_key=SecretStr("testkey"),
            minio_secret_key=SecretStr("testsecret"),
            minio_secure=False,
        )

        with patch("app.storage.minio_storage.Minio") as MockMinio:
            MinioStorage()
            MockMinio.assert_called_once_with(
                endpoint="192.168.31.104:9000",
                access_key="testkey",
                secret_key="testsecret",
                secure=False,
            )

    @patch("app.storage.minio_storage.settings")
    def test_missing_minio_config_raises_clear_error(self, mock_settings):
        mock_settings.storage = StorageSettings(
            provider="minio",
            minio_endpoint="",
            minio_bucket="",
            minio_access_key=SecretStr(""),
            minio_secret_key=SecretStr(""),
            minio_secure=False,
        )

        with pytest.raises(RuntimeError, match="MINIO_ENDPOINT"):
            MinioStorage()

    @patch("app.storage.minio_storage.settings")
    def test_empty_bucket_rejected_before_download(self, mock_settings):
        mock_settings.storage = StorageSettings(
            provider="minio",
            minio_endpoint="192.168.31.104:9000",
            minio_bucket="",
            minio_access_key=SecretStr("testkey"),
            minio_secret_key=SecretStr("testsecret"),
            minio_secure=False,
        )

        with patch("app.storage.minio_storage.Minio") as MockMinio:
            mock_client = MagicMock()
            MockMinio.return_value = mock_client

            storage = MinioStorage()
            with pytest.raises(RuntimeError, match="bucket is not configured"):
                storage.download_to_temp(bucket="", object_key="doc.txt")

        mock_client.fget_object.assert_not_called()


class TestGetStorageFactory:
    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported storage provider"):
            get_storage("azure")
