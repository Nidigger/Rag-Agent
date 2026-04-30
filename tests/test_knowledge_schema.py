"""Tests for app/schemas/knowledge.py — IngestRequest validation."""

import pytest
from pydantic import ValidationError

from app.schemas.knowledge import IngestRequest, StorageObject


class TestStorageObject:
    def test_minio_provider_valid(self):
        obj = StorageObject(object_key="original/default/kb/doc/v1/file.pdf")
        assert obj.provider == "minio"

    def test_explicit_provider(self):
        obj = StorageObject(
            provider="minio",
            object_key="key",
            bucket="bucket",
            file_name="file.pdf",
        )
        assert obj.provider == "minio"
        assert obj.bucket == "bucket"
        assert obj.file_name == "file.pdf"

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError):
            StorageObject(provider="s3", object_key="key")

    def test_empty_object_key_rejected(self):
        with pytest.raises(ValidationError):
            StorageObject(object_key="")

    def test_empty_bucket_rejected(self):
        with pytest.raises(ValidationError):
            StorageObject(bucket="", object_key="key")


class TestIngestRequest:
    def test_file_path_only_valid(self):
        req = IngestRequest(file_path="data/test.pdf")
        assert req.file_path == "data/test.pdf"
        assert req.storage is None

    def test_storage_only_valid(self):
        req = IngestRequest(
            storage=StorageObject(
                provider="minio",
                object_key="original/default/kb/doc/v1/file.pdf",
            )
        )
        assert req.storage is not None
        assert req.file_path is None

    def test_both_file_path_and_storage_rejected(self):
        with pytest.raises(ValidationError, match="Exactly one"):
            IngestRequest(
                file_path="data/test.pdf",
                storage=StorageObject(object_key="key"),
            )

    def test_neither_file_path_nor_storage_rejected(self):
        with pytest.raises(ValidationError, match="Exactly one"):
            IngestRequest()

    def test_defaults(self):
        req = IngestRequest(file_path="data/test.pdf")
        assert req.document_id is None
        assert req.document_version_id is None
        assert req.file_hash is None
        assert req.knowledge_base_id == "kb_default"
        assert req.tenant_id == "default"
