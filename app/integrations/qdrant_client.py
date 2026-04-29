"""Qdrant client module — connection management and low-level operations.

Provides a singleton QdrantClient that handles connection lifecycle,
collection management, and payload index creation. Higher-level modules
should use QdrantVectorStore (app/rag/qdrant_vector_store.py) which
wraps this client with the VectorStore interface.
"""

import logging

from qdrant_client import QdrantClient as QdrantSDKClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings

logger = logging.getLogger("rag-agent.qdrant_client")

_qdrant_client: QdrantSDKClient | None = None


def _load_api_key() -> str | None:
    """Load Qdrant API key from environment or .env file.

    Never returns the key in logs. The key should be set via:
        QDRANT_API_KEY=your-key
    """
    import os
    from pathlib import Path
    from dotenv import dotenv_values

    if "QDRANT_API_KEY" in os.environ:
        return os.environ["QDRANT_API_KEY"]

    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        dotenv = dotenv_values(env_file)
        return dotenv.get("QDRANT_API_KEY")

    return None


def get_qdrant_client() -> QdrantSDKClient:
    """Get or create the Qdrant client singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        api_key = _load_api_key()
        timeout = settings.vector.qdrant.timeout_seconds

        logger.info(
            "[qdrant_client] Connecting to Qdrant at %s (timeout=%ds)",
            settings.vector.qdrant.url,
            timeout,
        )

        _qdrant_client = QdrantSDKClient(
            url=settings.vector.qdrant.url,
            api_key=api_key,
            timeout=timeout,
        )
    return _qdrant_client


def reset_qdrant_client() -> None:
    """Reset the singleton client (used in tests)."""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
    _qdrant_client = None


def ensure_qdrant_collection(
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine",
) -> None:
    """Create the Qdrant collection if it does not exist.

    Safe to call on every startup — uses exists() check.
    """
    client = get_qdrant_client()

    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
    }
    distance_enum = distance_map.get(distance, Distance.COSINE)

    if client.collection_exists(collection_name):
        logger.info(
            "[qdrant_client] Collection '%s' already exists, skipping creation",
            collection_name,
        )
        return

    logger.info(
        "[qdrant_client] Creating collection '%s' (size=%d, distance=%s)",
        collection_name,
        vector_size,
        distance,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_enum),
    )

    _create_payload_indexes(collection_name)


def _create_payload_indexes(collection_name: str) -> None:
    """Create payload indexes for efficient filtering."""
    client = get_qdrant_client()
    index_fields = [
        "document_id",
        "document_version_id",
        "knowledge_base_id",
        "tenant_id",
        "enabled",
        "file_hash",
    ]
    for field in index_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",
            )
        except Exception as e:
            logger.debug(
                "[qdrant_client] Index on '%s' may already exist: %s", field, e
            )
    logger.info(
        "[qdrant_client] Payload indexes ensured for %d fields", len(index_fields)
    )
