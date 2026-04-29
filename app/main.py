"""FastAPI application factory.

Creates and configures the application with:
- Logging setup
- CORS and timing middleware
- Global exception handlers
- API v1 routes
- Vector store collection initialization on startup
"""

import logging

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.config import settings
from app.common.exceptions import register_exception_handlers
from app.middleware.http import register_middleware
from app.observability.logging import setup_logging

logger = logging.getLogger("rag-agent.main")


def _init_vector_store():
    """Ensure the configured vector store collection exists at startup."""
    try:
        provider = settings.vector.provider
        if provider == "qdrant":
            from app.rag.qdrant_vector_store import QdrantVectorStore

            store = QdrantVectorStore()
            store.ensure_collection()
            logger.info(
                "[main] Qdrant collection '%s' ready",
                store.collection_name,
            )
        else:
            from app.rag.vector_store import ChromaVectorStore

            store = ChromaVectorStore()
            store.ensure_collection()
            logger.info("[main] Chroma vector store ready")
    except Exception as e:
        logger.warning(
            "[main] Vector store initialization skipped: %s", e
        )


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.server.project_name,
        version=settings.server.version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    register_middleware(app)
    register_exception_handlers(app)

    app.include_router(api_v1_router, prefix=settings.server.api_v1_prefix)

    # Initialize vector store collection on startup
    _init_vector_store()

    return app


app = create_app()
