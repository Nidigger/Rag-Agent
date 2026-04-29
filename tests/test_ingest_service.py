"""Unit tests for IngestService."""

import hashlib
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.rag.ingest_service import IngestService, _compute_file_hash
from app.rag.vector_store_base import VectorChunk, VectorStore


class FakeVectorStore(VectorStore):
    """In-memory fake vector store for testing IngestService."""

    def __init__(self):
        self.chunks: list[VectorChunk] = []
        self.deleted: list[str] = []
        self.disabled: list[str] = []

    def ensure_collection(self) -> None:
        pass

    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query, top_k, filters=None):
        return []

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)

    def disable_document(self, document_id: str) -> None:
        self.disabled.append(document_id)

    def health_check(self) -> bool:
        return True


class TestIngestService:
    def setup_method(self):
        self.store = FakeVectorStore()
        self.service = IngestService(self.store)

    def test_ingest_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.service.ingest_file(
                file_path="/nonexistent/file.txt",
                document_id="doc_001",
            )

    def test_ingest_txt_file_success(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("这是第一段测试内容。需要足够长才能被切分成多个chunk。" * 20)
            tmp.close()

            result = self.service.ingest_file(
                file_path=tmp.name,
                document_id="doc_001",
                document_version_id="ver_001",
                knowledge_base_id="kb_test",
            )

            assert result["document_id"] == "doc_001"
            assert result["chunk_count"] > 0
            assert result["status"] == "success"
            assert len(self.store.chunks) == result["chunk_count"]

            for chunk in self.store.chunks:
                assert chunk.document_id == "doc_001"
                assert chunk.document_version_id == "ver_001"
                assert chunk.knowledge_base_id == "kb_test"
                assert chunk.tenant_id == "default"
                assert chunk.enabled is True
                assert len(chunk.point_id) == 36
                uuid.UUID(chunk.point_id)
                assert len(chunk.content_hash) == 64
        finally:
            os.unlink(tmp.name)

    def test_ingest_txt_file_with_custom_tenant(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("测试内容用于租户隔离。" * 30)
            tmp.close()

            self.service.ingest_file(
                file_path=tmp.name,
                document_id="doc_002",
                tenant_id="tenant_custom",
                knowledge_base_id="kb_custom",
            )

            for chunk in self.store.chunks:
                assert chunk.tenant_id == "tenant_custom"
                assert chunk.knowledge_base_id == "kb_custom"
        finally:
            os.unlink(tmp.name)

    def test_ingest_file_hash_computed(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("用于测试文件哈希的内容。" * 20)
            tmp.close()

            self.service.ingest_file(
                file_path=tmp.name,
                document_id="doc_003",
            )

            for chunk in self.store.chunks:
                assert chunk.file_hash is not None
                assert len(chunk.file_hash) == 64
                assert chunk.file_type == "txt"
        finally:
            os.unlink(tmp.name)

    def test_ingest_provided_file_hash(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("预提供哈希的测试内容。" * 20)
            tmp.close()

            self.service.ingest_file(
                file_path=tmp.name,
                document_id="doc_004",
                file_hash="abc123def456",
            )

            for chunk in self.store.chunks:
                assert chunk.file_hash == "abc123def456"
        finally:
            os.unlink(tmp.name)

    def test_ingest_point_id_stable(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            content = "稳定ID测试内容。" * 20
            tmp.write(content)
            tmp.close()

            store1 = FakeVectorStore()
            service1 = IngestService(store1)
            service1.ingest_file(file_path=tmp.name, document_id="doc_stable")

            store2 = FakeVectorStore()
            service2 = IngestService(store2)
            service2.ingest_file(file_path=tmp.name, document_id="doc_stable")

            ids1 = [c.point_id for c in store1.chunks]
            ids2 = [c.point_id for c in store2.chunks]
            assert ids1 == ids2
        finally:
            os.unlink(tmp.name)

    def test_delete_document_vectors(self):
        result = self.service.delete_document_vectors("doc_delete")
        assert result["document_id"] == "doc_delete"
        assert result["status"] == "deleted"
        assert "doc_delete" in self.store.deleted

    def test_disable_document(self):
        result = self.service.disable_document("doc_disable")
        assert result["document_id"] == "doc_disable"
        assert result["status"] == "disabled"
        assert "doc_disable" in self.store.disabled

    def test_ingest_empty_file_raises(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("")
            tmp.close()

            with pytest.raises(ValueError):
                self.service.ingest_file(
                    file_path=tmp.name,
                    document_id="doc_empty",
                )
        finally:
            os.unlink(tmp.name)

    def test_ingest_unsupported_file_type_raises(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".exe", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("binary content")
            tmp.close()

            with pytest.raises(ValueError):
                self.service.ingest_file(
                    file_path=tmp.name,
                    document_id="doc_bad",
                )
        finally:
            os.unlink(tmp.name)

    def test_source_auto_from_filename(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            tmp.write("source测试。" * 20)
            tmp.close()

            self.service.ingest_file(
                file_path=tmp.name,
                document_id="doc_source",
            )

            basename = os.path.basename(tmp.name)
            for chunk in self.store.chunks:
                assert chunk.source == basename
        finally:
            os.unlink(tmp.name)


class TestComputeFileHash:
    def test_known_content_hash(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        )
        try:
            tmp.write(b"hello world")
            tmp.close()

            result = _compute_file_hash(tmp.name)
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert result == expected
        finally:
            os.unlink(tmp.name)

    def test_empty_file_hash(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        )
        try:
            tmp.write(b"")
            tmp.close()

            result = _compute_file_hash(tmp.name)
            expected = hashlib.sha256(b"").hexdigest()
            assert result == expected
        finally:
            os.unlink(tmp.name)
