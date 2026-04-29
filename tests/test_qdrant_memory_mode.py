"""Integration tests for QdrantVectorStore using real Qdrant memory mode.

These tests verify the actual qdrant-client SDK behaviour, not mocks.
Uses QdrantClient(":memory:") to avoid needing an external Qdrant server.
"""

import uuid as uuid_lib
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.rag.qdrant_vector_store import QdrantVectorStore
from app.rag.vector_store_base import (
    VectorChunk,
    generate_content_hash,
    generate_point_id,
)


@pytest.fixture
def memory_store():
    """Create a QdrantVectorStore backed by an in-memory Qdrant client."""
    client = QdrantClient(":memory:")

    collection_name = "test_collection_mem"
    vector_size = 128

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    fake_embed_query = [0.1] * vector_size

    with patch("app.rag.qdrant_vector_store.get_qdrant_client", return_value=client), \
         patch("app.rag.qdrant_vector_store.get_embed_model") as mock_embed:
        mock_embed.return_value.embed_query.return_value = fake_embed_query
        store = QdrantVectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
        )
        yield store


class TestQdrantVectorStoreMemoryMode:
    def test_upsert_and_search_real_sdk(self, memory_store):
        point_id = generate_point_id("doc_001", 0, "content_hash_abc")
        chunk = VectorChunk(
            point_id=point_id,
            document_id="doc_001",
            chunk_index=0,
            content="this is a test chunk about cleaning robots",
            content_hash="content_hash_abc",
            knowledge_base_id="kb_default",
        )
        memory_store.upsert_chunks([chunk])

        results = memory_store.search(
            "how to clean robot",
            top_k=3,
            filters={"enabled": True, "knowledge_base_id": "kb_default"},
        )

        assert len(results) > 0
        assert results[0].document_id == "doc_001"
        assert results[0].content == "this is a test chunk about cleaning robots"

    def test_point_id_is_valid_uuid(self, memory_store):
        point_id = generate_point_id("doc_001", 0, "hash123")
        uuid_lib.UUID(point_id)

    def test_disable_then_search_excludes(self, memory_store):
        point_id = generate_point_id("doc_002", 0, "hash_disabled")
        chunk = VectorChunk(
            point_id=point_id,
            document_id="doc_002",
            chunk_index=0,
            content="disabled document chunk",
            content_hash="hash_disabled",
            knowledge_base_id="kb_default",
        )
        memory_store.upsert_chunks([chunk])

        results_before = memory_store.search(
            "disabled document",
            top_k=3,
            filters={"enabled": True, "knowledge_base_id": "kb_default"},
        )
        assert len(results_before) > 0

        memory_store.disable_document("doc_002")

        results_after = memory_store.search(
            "disabled document",
            top_k=3,
            filters={"enabled": True, "knowledge_base_id": "kb_default"},
        )
        assert len(results_after) == 0

    def test_delete_document_removes_points(self, memory_store):
        point_id = generate_point_id("doc_003", 0, "hash_delete")
        chunk = VectorChunk(
            point_id=point_id,
            document_id="doc_003",
            chunk_index=0,
            content="will be deleted",
            content_hash="hash_delete",
            knowledge_base_id="kb_default",
        )
        memory_store.upsert_chunks([chunk])

        results_before = memory_store.search("will be deleted", top_k=3)
        assert len(results_before) > 0

        memory_store.delete_document("doc_003")

        results_after = memory_store.search("will be deleted", top_k=3)
        assert len(results_after) == 0

    def test_upsert_same_point_id_no_duplicates(self, memory_store):
        point_id = generate_point_id("doc_dup", 0, "hash_dup")
        chunk = VectorChunk(
            point_id=point_id,
            document_id="doc_dup",
            chunk_index=0,
            content="original content",
            content_hash="hash_dup",
            knowledge_base_id="kb_default",
        )
        memory_store.upsert_chunks([chunk])

        chunk2 = VectorChunk(
            point_id=point_id,
            document_id="doc_dup",
            chunk_index=0,
            content="updated content",
            content_hash="hash_dup",
            knowledge_base_id="kb_default",
        )
        memory_store.upsert_chunks([chunk2])

        results = memory_store.search("content", top_k=10)
        doc_results = [r for r in results if r.document_id == "doc_dup"]
        assert len(doc_results) == 1
        assert doc_results[0].content == "updated content"

    def test_health_check_healthy(self, memory_store):
        assert memory_store.health_check() is True
