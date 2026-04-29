"""Vector store health check API endpoint."""

import logging

from fastapi import APIRouter

from app.common.response import success
from app.config import settings
from app.rag.vector_store_base import VectorStore
from app.rag.vector_store import ChromaVectorStore
from app.rag.qdrant_vector_store import QdrantVectorStore

logger = logging.getLogger("rag-agent.vector_api")
router = APIRouter()


def _get_vector_store() -> VectorStore:
    if settings.vector.provider == "qdrant":
        return QdrantVectorStore()
    return ChromaVectorStore()


@router.get("/health")
async def vector_health_check():
    provider = settings.vector.provider
    store = _get_vector_store()

    try:
        healthy = store.health_check()
        status = "healthy" if healthy else "unhealthy"
    except Exception as e:
        logger.error("[vector/health] Health check error: %s", e)
        status = "unhealthy"

    collection = (
        settings.vector.qdrant.collection_name
        if provider == "qdrant"
        else settings.chroma.collection_name
    )

    return success(data={
        "provider": provider,
        "status": status,
        "collection": collection,
    })
