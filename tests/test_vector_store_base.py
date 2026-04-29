"""Unit tests for vector_store_base models and utilities."""

import uuid as uuid_lib

import pytest

from app.rag.vector_store_base import (
    RetrievedChunk,
    VectorChunk,
    generate_content_hash,
    generate_point_id,
)


class TestVectorChunk:
    def test_basic_construction(self):
        chunk = VectorChunk(
            point_id="11111111-1111-1111-1111-111111111111",
            document_id="doc_001",
            chunk_index=0,
            content="hello world",
            content_hash="hash123",
        )
        assert chunk.document_id == "doc_001"
        assert chunk.chunk_index == 0
        assert chunk.content == "hello world"
        assert chunk.enabled is True
        assert chunk.tenant_id == "default"
        assert chunk.knowledge_base_id == "kb_default"

    def test_optional_fields(self):
        chunk = VectorChunk(
            point_id="11111111-1111-1111-1111-111111111111",
            document_id="doc_001",
            chunk_index=5,
            content="test content",
            content_hash="hash456",
            document_version_id="ver_001",
            source="manual.pdf",
            file_type="pdf",
            file_hash="sha256...",
            enabled=False,
            tenant_id="tenant_001",
            knowledge_base_id="kb_001",
            metadata={"page": 3},
        )
        assert chunk.document_version_id == "ver_001"
        assert chunk.source == "manual.pdf"
        assert chunk.file_type == "pdf"
        assert chunk.file_hash == "sha256..."
        assert chunk.enabled is False
        assert chunk.tenant_id == "tenant_001"
        assert chunk.knowledge_base_id == "kb_001"
        assert chunk.metadata == {"page": 3}

    def test_model_dump(self):
        chunk = VectorChunk(
            point_id="11111111-1111-1111-1111-111111111111",
            document_id="doc_001",
            chunk_index=0,
            content="hello",
            content_hash="hash789",
        )
        data = chunk.model_dump()
        assert data["point_id"] == "11111111-1111-1111-1111-111111111111"
        assert data["document_id"] == "doc_001"


class TestRetrievedChunk:
    def test_basic_construction(self):
        chunk = RetrievedChunk(
            point_id="11111111-1111-1111-1111-111111111111",
            document_id="doc_001",
            chunk_index=0,
            content="hello world",
            score=0.95,
        )
        assert chunk.document_id == "doc_001"
        assert chunk.chunk_index == 0
        assert chunk.content == "hello world"
        assert chunk.score == 0.95
        assert chunk.metadata == {}

    def test_with_metadata(self):
        chunk = RetrievedChunk(
            point_id="11111111-1111-1111-1111-111111111111",
            document_id="doc_002",
            chunk_index=3,
            content="content",
            score=0.88,
            metadata={"source": "manual.pdf"},
        )
        assert chunk.metadata["source"] == "manual.pdf"


class TestGeneratePointId:
    def test_same_input_produces_same_id(self):
        id1 = generate_point_id("doc_001", 0, "hash123")
        id2 = generate_point_id("doc_001", 0, "hash123")
        assert id1 == id2

    def test_different_document_id_produces_different_id(self):
        id1 = generate_point_id("doc_001", 0, "hash123")
        id2 = generate_point_id("doc_002", 0, "hash123")
        assert id1 != id2

    def test_different_chunk_index_produces_different_id(self):
        id1 = generate_point_id("doc_001", 0, "hash123")
        id2 = generate_point_id("doc_001", 1, "hash123")
        assert id1 != id2

    def test_different_content_hash_produces_different_id(self):
        id1 = generate_point_id("doc_001", 0, "hash123")
        id2 = generate_point_id("doc_001", 0, "hash456")
        assert id1 != id2

    def test_output_is_valid_uuid(self):
        result = generate_point_id("doc_001", 0, "hash123")
        parsed = uuid_lib.UUID(result)
        assert isinstance(parsed, uuid_lib.UUID)
        assert len(str(parsed)) == 36

    def test_output_is_stable_across_calls(self):
        id1 = generate_point_id("doc_stable", 0, "content_hash_abc")
        id2 = generate_point_id("doc_stable", 0, "content_hash_abc")
        assert id1 == id2


class TestGenerateContentHash:
    def test_same_content_produces_same_hash(self):
        h1 = generate_content_hash("hello world")
        h2 = generate_content_hash("hello world")
        assert h1 == h2

    def test_different_content_produces_different_hash(self):
        h1 = generate_content_hash("hello world")
        h2 = generate_content_hash("hello world!")
        assert h1 != h2

    def test_output_is_stable_hex_string(self):
        result = generate_content_hash("test")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
