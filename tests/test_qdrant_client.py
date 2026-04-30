"""Unit tests for app/integrations/qdrant_client.py.

Validates that:
- ensure_qdrant_collection creates collection when it does not exist.
- ensure_qdrant_collection skips creation when already exists and size matches.
- ensure_qdrant_collection raises RuntimeError on vector size mismatch.
"""

import pytest
from unittest.mock import MagicMock, patch

from qdrant_client.models import Distance


def _mock_collection_info_single(vector_size: int) -> MagicMock:
    """Create a MagicMock that behaves like a CollectionInfo with single vector."""
    mock = MagicMock()
    mock.config.params.vectors.size = vector_size
    return mock


class TestEnsureCollectionCreation:
    @patch("app.integrations.qdrant_client.get_qdrant_client")
    def test_creates_collection_when_not_exists(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        from app.integrations.qdrant_client import ensure_qdrant_collection

        ensure_qdrant_collection(
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
        )

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"

        vectors_config = call_kwargs["vectors_config"]
        assert vectors_config.size == 1536
        assert vectors_config.distance == Distance.COSINE

    @patch("app.integrations.qdrant_client.get_qdrant_client")
    def test_creates_payload_indexes(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        from app.integrations.qdrant_client import ensure_qdrant_collection

        ensure_qdrant_collection(
            collection_name="test_collection",
            vector_size=128,
        )

        assert mock_client.create_payload_index.call_count >= 1


class TestEnsureCollectionExistsMatching:
    @patch("app.integrations.qdrant_client.get_qdrant_client")
    def test_skips_creation_when_exists_and_size_matches(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _mock_collection_info_single(1536)
        mock_get_client.return_value = mock_client

        from app.integrations.qdrant_client import ensure_qdrant_collection

        ensure_qdrant_collection(
            collection_name="my_collection",
            vector_size=1536,
        )

        mock_client.create_collection.assert_not_called()


class TestEnsureCollectionMismatch:
    @patch("app.integrations.qdrant_client.get_qdrant_client")
    def test_raises_on_vector_size_mismatch(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _mock_collection_info_single(1024)
        mock_get_client.return_value = mock_client

        from app.integrations.qdrant_client import ensure_qdrant_collection

        with pytest.raises(RuntimeError, match="vector size mismatch"):
            ensure_qdrant_collection(
                collection_name="my_collection",
                vector_size=1536,
            )

    @patch("app.integrations.qdrant_client.get_qdrant_client")
    def test_mismatch_error_message_includes_diagnostics(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = _mock_collection_info_single(1024)
        mock_get_client.return_value = mock_client

        from app.integrations.qdrant_client import ensure_qdrant_collection

        with pytest.raises(RuntimeError) as exc_info:
            ensure_qdrant_collection(
                collection_name="my_collection",
                vector_size=1536,
            )

        error_msg = str(exc_info.value)
        assert "configured=1536" in error_msg
        assert "existing=1024" in error_msg
        assert "recreate collection" in error_msg.lower()
