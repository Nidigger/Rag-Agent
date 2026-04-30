"""Document ingest service.

Orchestrates the full pipeline:
  read file -> split into chunks -> generate embeddings -> upsert to vector store.

Supports both Qdrant (primary) and Chroma (fallback) providers via the
VectorStore abstract interface.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.vector_store_base import (
    VectorChunk,
    VectorStore,
    generate_content_hash,
    generate_point_id,
)
from app.utils.file_handler import pdf_loader, txt_loader

logger = logging.getLogger("rag-agent.ingest_service")


class IngestService:
    """Service for ingesting documents into the vector store."""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.vector.ingest.chunk_size,
            chunk_overlap=settings.vector.ingest.chunk_overlap,
            separators=[
                "\n\n", "\n", ".", "!", "?",
                "\u3002", "\uff01", "\uff1f",
                " ", "",
            ],
            length_function=len,
        )

    def ingest_file(
        self,
        file_path: str,
        document_id: str,
        document_version_id: Optional[str] = None,
        file_hash: Optional[str] = None,
        knowledge_base_id: str = "kb_default",
        tenant_id: str = "default",
        source: Optional[str] = None,
    ) -> dict:
        """Ingest a single file into the vector store.

        Args:
            file_path: Absolute or relative path to the document.
            document_id: Stable document identifier.
            document_version_id: Version identifier for versioning support.
            file_hash: Pre-computed SHA256 hash of the file.
            knowledge_base_id: Knowledge base scope.
            tenant_id: Tenant scope for multi-tenancy.
            source: Source filename for metadata.

        Returns:
            Dict with document_id, chunk_count, and status.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        documents = self._load_file(file_path)
        if not documents:
            raise ValueError(f"No valid content extracted from: {file_path}")

        if source is None:
            source = os.path.basename(file_path)

        if file_hash is None:
            file_hash = _compute_file_hash(file_path)

        file_type = Path(file_path).suffix.lower().lstrip(".")

        chunks = self._splitter.split_documents(documents)
        if not chunks:
            raise ValueError(f"No chunks produced for: {file_path}")

        vector_chunks: list[VectorChunk] = []
        for i, chunk in enumerate(chunks):
            content_hash = generate_content_hash(chunk.page_content)
            point_id = generate_point_id(document_id, i, content_hash)

            vector_chunks.append(
                VectorChunk(
                    point_id=point_id,
                    document_id=document_id,
                    document_version_id=document_version_id,
                    chunk_index=i,
                    content=chunk.page_content,
                    content_hash=content_hash,
                    source=source,
                    file_type=file_type,
                    file_hash=file_hash,
                    enabled=True,
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                    metadata=chunk.metadata,
                )
            )

        batch_size = settings.vector.ingest.batch_size
        for batch_start in range(0, len(vector_chunks), batch_size):
            batch = vector_chunks[batch_start:batch_start + batch_size]
            self._vector_store.upsert_chunks(batch)
            logger.info(
                "[ingest_service] Batch %d-%d/%d ingested",
                batch_start,
                batch_start + len(batch),
                len(vector_chunks),
            )

        chunks_detail = []
        version_id = document_version_id or "0"
        collection = self._vector_store.collection_name
        for vc in vector_chunks:
            chunks_detail.append({
                "chunk_id": f"{document_id}:{version_id}:{vc.chunk_index}",
                "qdrant_collection": collection,
                "qdrant_point_id": vc.point_id,
                "content_hash": vc.content_hash,
            })

        return {
            "document_id": document_id,
            "chunk_count": len(vector_chunks),
            "status": "success",
            "chunks": chunks_detail,
        }

    def delete_document_vectors(self, document_id: str) -> dict:
        """Delete all vectors for a given document."""
        self._vector_store.delete_document(document_id)
        return {
            "document_id": document_id,
            "status": "deleted",
        }

    def disable_document(self, document_id: str) -> dict:
        """Logically disable a document (set enabled=false)."""
        if hasattr(self._vector_store, "disable_document"):
            self._vector_store.disable_document(document_id)
        else:
            self._vector_store.delete_document(document_id)

        return {
            "document_id": document_id,
            "status": "disabled",
        }

    def _load_file(self, file_path: str):
        """Load file content based on extension."""
        if file_path.endswith(".txt"):
            return txt_loader(file_path)
        if file_path.endswith(".pdf"):
            return pdf_loader(file_path)
        raise ValueError(f"Unsupported file type: {file_path}")


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
