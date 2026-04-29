"""Unit tests for QdrantVectorStore using mocked Qdrant client."""

import uuid as uuid_lib
from unittest.mock import MagicMock, patch, call

from qdrant_client.models import ScoredPoint, Distance, VectorParams, QueryResponse

from app.rag.qdrant_vector_store import QdrantVectorStore
from app.rag.vector_store_base import VectorChunk, RetrievedChunk


VALID_UUID_1 = "11111111-1111-1111-1111-111111111111"
VALID_UUID_2 = "22222222-2222-2222-2222-222222222222"


class TestQdrantVectorStoreUpsert:
    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_upsert_chunks_calls_client(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = mock_embed

        store = QdrantVectorStore(
            collection_name="test_collection",
            vector_size=3,
        )

        chunks = [
            VectorChunk(
                point_id=VALID_UUID_1,
                document_id="doc_001",
                chunk_index=0,
                content="chunk text",
                content_hash="hash_001",
            )
        ]

        store.upsert_chunks(chunks)

        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["wait"] is True
        assert len(call_args[1]["points"]) == 1
        assert call_args[1]["points"][0].id == uuid_lib.UUID(VALID_UUID_1)

    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_upsert_empty_chunks_no_call(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = MagicMock()

        store = QdrantVectorStore()
        store.upsert_chunks([])

        mock_client.upsert.assert_not_called()

    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_upsert_includes_payload_fields(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2]
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = mock_embed

        store = QdrantVectorStore(collection_name="test", vector_size=2)

        chunks = [
            VectorChunk(
                point_id=VALID_UUID_1,
                document_id="doc_001",
                chunk_index=3,
                content="text content",
                content_hash="hash_content",
                document_version_id="ver_001",
                source="manual.pdf",
                file_type="pdf",
                file_hash="file_hash_abc",
                tenant_id="tenant_x",
                knowledge_base_id="kb_x",
            )
        ]

        store.upsert_chunks(chunks)

        payload = mock_client.upsert.call_args[1]["points"][0].payload
        assert payload["document_id"] == "doc_001"
        assert payload["chunk_index"] == 3
        assert payload["content"] == "text content"
        assert payload["content_hash"] == "hash_content"
        assert payload["document_version_id"] == "ver_001"
        assert payload["source"] == "manual.pdf"
        assert payload["file_type"] == "pdf"
        assert payload["file_hash"] == "file_hash_abc"
        assert payload["tenant_id"] == "tenant_x"
        assert payload["knowledge_base_id"] == "kb_x"
        assert payload["enabled"] is True


class TestQdrantVectorStoreSearch:
    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_search_returns_retrieved_chunks(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = mock_embed

        mock_response = MagicMock()
        mock_response.points = [
            ScoredPoint(
                id=VALID_UUID_1,
                version=0,
                score=0.95,
                vector=None,
                payload={
                    "document_id": "doc_001",
                    "chunk_index": 0,
                    "content": "relevant chunk text",
                },
            ),
            ScoredPoint(
                id=VALID_UUID_2,
                version=0,
                score=0.80,
                vector=None,
                payload={
                    "document_id": "doc_002",
                    "chunk_index": 1,
                    "content": "another chunk",
                },
            ),
        ]
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(
            collection_name="test",
            vector_size=3,
        )

        results = store.search("test query", top_k=3)

        assert len(results) == 2
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].document_id == "doc_001"
        assert results[0].content == "relevant chunk text"
        assert results[0].score == 0.95
        assert results[1].document_id == "doc_002"
        assert results[1].score == 0.80

    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_search_with_filters(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2]
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = mock_embed

        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(collection_name="test", vector_size=2)
        store.search(
            "query",
            top_k=5,
            filters={"enabled": True, "knowledge_base_id": "kb_default"},
        )

        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["query_filter"] is not None
        assert len(call_kwargs["query_filter"].must) == 2

    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_search_as_documents(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1, 0.2]
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = mock_embed

        mock_response = MagicMock()
        mock_response.points = [
            ScoredPoint(
                id=VALID_UUID_1,
                version=0,
                score=0.90,
                vector=None,
                payload={
                    "document_id": "doc_001",
                    "chunk_index": 0,
                    "content": "chunk text",
                },
            )
        ]
        mock_client.query_points.return_value = mock_response

        store = QdrantVectorStore(collection_name="test", vector_size=2)
        docs = store.search_as_documents("query", top_k=3)

        assert len(docs) == 1
        assert docs[0].page_content == "chunk text"
        assert docs[0].metadata["document_id"] == "doc_001"
        assert docs[0].metadata["score"] == 0.90


class TestQdrantVectorStoreDelete:
    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_delete_document(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = MagicMock()

        store = QdrantVectorStore(collection_name="test")
        store.delete_document("doc_001")

        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args[1]
        assert call_kwargs["collection_name"] == "test"
        assert len(call_kwargs["points_selector"].must) == 1
        assert call_kwargs["points_selector"].must[0].key == "document_id"

    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_disable_document(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = MagicMock()

        store = QdrantVectorStore(collection_name="test")
        store.disable_document("doc_001")

        mock_client.set_payload.assert_called_once()
        call_kwargs = mock_client.set_payload.call_args[1]
        assert call_kwargs["payload"] == {"enabled": False}
        assert call_kwargs["collection_name"] == "test"
        assert call_kwargs["wait"] is True
        assert call_kwargs["points"] is not None


class TestQdrantVectorStoreHealthCheck:
    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_health_check_healthy(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = MagicMock()

        store = QdrantVectorStore(collection_name="test")
        assert store.health_check() is True

    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_health_check_unhealthy(self, mock_get_client, mock_get_embed):
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = RuntimeError("Connection refused")
        mock_get_client.return_value = mock_client
        mock_get_embed.return_value = MagicMock()

        store = QdrantVectorStore(collection_name="test")
        assert store.health_check() is False


class TestQdrantVectorStoreEnsureCollection:
    @patch("app.rag.qdrant_vector_store.ensure_qdrant_collection")
    @patch("app.rag.qdrant_vector_store.get_embed_model")
    @patch("app.rag.qdrant_vector_store.get_qdrant_client")
    def test_ensure_collection_calls_helper(
        self, mock_get_client, mock_get_embed, mock_ensure
    ):
        mock_get_client.return_value = MagicMock()
        mock_get_embed.return_value = MagicMock()

        store = QdrantVectorStore(
            collection_name="my_collection",
            vector_size=1536,
            distance="Cosine",
        )
        store.ensure_collection()

        mock_ensure.assert_called_once_with(
            collection_name="my_collection",
            vector_size=1536,
            distance="Cosine",
        )
