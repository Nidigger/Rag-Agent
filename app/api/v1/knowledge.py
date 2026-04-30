"""Knowledge base management API endpoints.

Provides internal endpoints for document ingest, delete, and rebuild.
In the current phase these are accessible via developer token; future
phases will restrict access to Spring Boot internal calls only.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Header, Request

from app.common.exceptions import (
    DocumentNotFoundError,
    IngestFailedError,
    UnauthorizedError,
    VectorStoreUnavailableError,
)
from app.common.response import ErrorCode, error, success
from app.config import settings
from app.rag.ingest_service import IngestService
from app.rag.vector_store_base import VectorStore
from app.rag.vector_store import ChromaVectorStore
from app.rag.qdrant_vector_store import QdrantVectorStore
from app.schemas.knowledge import (
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    IngestRequest,
    IngestResponse,
)
from app.storage import get_storage

logger = logging.getLogger("rag-agent.knowledge_api")
router = APIRouter()


def _get_vector_store() -> VectorStore:
    if settings.vector.provider == "qdrant":
        store = QdrantVectorStore()
        store.ensure_collection()
        return store
    return ChromaVectorStore()


def _verify_internal_token(x_internal_token: str | None = Header(None)) -> str:
    """Verify the X-Internal-Token header for internal API access.

    Reads the configured token from settings.security.internal_token
    (loaded via the unified config loader from .env / OS env).

    In debug mode, a warning is logged but access is still allowed
    for local development convenience.
    """
    token_secret = settings.security.internal_token
    expected = token_secret.get_secret_value() if token_secret else None

    if not expected:
        if settings.server.debug:
            logger.warning(
                "[knowledge_api] FASTAPI_INTERNAL_TOKEN not configured — "
                "allowing access in DEBUG mode only"
            )
            return "debug"
        raise UnauthorizedError(
            "FASTAPI_INTERNAL_TOKEN is not configured on server"
        )

    if x_internal_token is None or x_internal_token != expected:
        raise UnauthorizedError("Invalid or missing X-Internal-Token")

    return x_internal_token


def _validate_file_path(file_path: str) -> None:
    """Validate that a file path is safe to access.

    - Rejects path traversal attempts (..).
    - Only allows paths within the project root or configured data directory.
    """
    if ".." in str(Path(file_path)):
        raise UnauthorizedError("Path traversal is not allowed")

    resolved = Path(file_path).resolve()
    project_root = settings.project_root.resolve()
    data_dir = (project_root / settings.chroma.data_path).resolve()

    try:
        resolved.relative_to(project_root)
    except ValueError:
        try:
            resolved.relative_to(data_dir)
        except ValueError:
            try:
                temp_dir = Path(tempfile.gettempdir()).resolve()
                resolved.relative_to(temp_dir)
            except ValueError:
                raise UnauthorizedError(
                    "File path must be within project root or data directory"
                )


@router.post("/documents/{document_id}/ingest")
async def ingest_document(
    document_id: str,
    req: IngestRequest,
    token: str = Depends(_verify_internal_token),
):
    """Ingest a document into the vector store.

    Supports two modes:
      - **file_path**: Local file path (validated for safety).
      - **storage**: Object storage reference (e.g. MinIO). The object is
        downloaded to a temporary file, processed, and cleaned up.

    Spring Boot not yet implemented — this endpoint serves as both
    the developer-facing and future Spring Boot internal endpoint.
    """
    local_ref = None
    try:
        vector_store = _get_vector_store()

        if not vector_store.health_check():
            raise VectorStoreUnavailableError()

        if req.file_path:
            _validate_file_path(req.file_path)
            file_path = req.file_path
            source = Path(req.file_path).name

            if not os.path.exists(file_path):
                raise DocumentNotFoundError(
                    f"File not found: {file_path}"
                )
        else:
            bucket = req.storage.bucket or settings.storage.minio_bucket
            if not bucket:
                raise IngestFailedError(
                    "MinIO bucket is not configured; pass storage.bucket "
                    "or set MINIO_BUCKET"
                )

            local_ref = get_storage(req.storage.provider).download_to_temp(
                bucket=bucket,
                object_key=req.storage.object_key,
            )
            file_path = local_ref.path
            source = req.storage.file_name or local_ref.filename

        ingest_service = IngestService(vector_store)

        result = await asyncio.to_thread(
            ingest_service.ingest_file,
            file_path=file_path,
            document_id=req.document_id or document_id,
            document_version_id=req.document_version_id,
            file_hash=req.file_hash,
            knowledge_base_id=req.knowledge_base_id,
            tenant_id=req.tenant_id,
            source=source,
        )

        logger.info(
            "[knowledge_api] Ingested document '%s': %d chunks",
            result["document_id"],
            result["chunk_count"],
        )

        return success(data=result)

    except (VectorStoreUnavailableError, DocumentNotFoundError, UnauthorizedError):
        raise
    except Exception as e:
        logger.error("[knowledge_api] Ingest failed: %s", e, exc_info=True)
        raise IngestFailedError(str(e))
    finally:
        if local_ref and local_ref.should_cleanup:
            Path(local_ref.path).unlink(missing_ok=True)


@router.delete("/documents/{document_id}/vectors")
async def delete_document_vectors(
    document_id: str,
    req: DeleteDocumentRequest | None = None,
    token: str = Depends(_verify_internal_token),
):
    """Delete all vectors for a given document.

    Supports logical (enabled=false) and physical deletion.
    """
    try:
        vector_store = _get_vector_store()

        if not vector_store.health_check():
            raise VectorStoreUnavailableError()

        ingest_service = IngestService(vector_store)

        if req and req.physical:
            result = await asyncio.to_thread(
                ingest_service.delete_document_vectors,
                document_id,
            )
        else:
            result = await asyncio.to_thread(
                ingest_service.disable_document,
                document_id,
            )

        logger.info(
            "[knowledge_api] Deleted/disabled document '%s': %s",
            document_id,
            result["status"],
        )

        return success(data=result)

    except (VectorStoreUnavailableError, UnauthorizedError):
        raise
    except Exception as e:
        logger.error(
            "[knowledge_api] Delete failed for '%s': %s", document_id, e, exc_info=True
        )
        raise IngestFailedError(str(e))
