"""Vector store abstract interface and shared data models.

Defines the contract that all vector store implementations must fulfill.
Business code depends on this interface, not on concrete implementations.
"""

import hashlib
import uuid as uuid_lib
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

POINT_ID_NAMESPACE = uuid_lib.UUID("1f6e8c3a-47b2-4d7e-9a1c-3f584b0e6d2a")


class VectorChunk(BaseModel):
    """A document chunk ready for vector store ingestion."""

    point_id: str = Field(..., description="Stable unique point identifier (UUID)")
    document_id: str = Field(..., description="Source document ID")
    document_version_id: str | None = None
    chunk_index: int = Field(..., description="0-based chunk index within document")
    content: str = Field(..., description="Chunk text content")
    content_hash: str = Field(..., description="SHA256 hash of content")
    source: str | None = None
    file_type: str | None = None
    file_hash: str | None = None
    enabled: bool = True
    tenant_id: str = "default"
    knowledge_base_id: str = "kb_default"
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector store search."""

    point_id: str
    document_id: str
    chunk_index: int
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract interface for vector store operations.

    All vector store implementations (Qdrant, Chroma, etc.) must
    implement this interface. Business layers depend only on this,
    enabling provider swaps without touching application logic.
    """

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create or verify the collection exists. Called at startup."""
        ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[VectorChunk]) -> None:
        """Insert or update vector chunks. Stable point_id prevents duplicates."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Search for similar chunks given a text query.

        Args:
            query: Natural language query text.
            top_k: Maximum number of results to return.
            filters: Payload filter dict for scoping results
                (e.g. {"enabled": true, "knowledge_base_id": "default"}).

        Returns:
            Ordered list of RetrievedChunk (highest score first).
        """
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document by document_id."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the vector store is reachable and operational."""
        ...


def generate_point_id(
    document_id: str,
    chunk_index: int,
    content_hash: str,
) -> str:
    """Generate a stable UUID point ID using uuid5.

    Uses a stable namespace UUID so that re-ingesting the same document
    with the same content produces the identical point ID — preventing
    duplicate vectors in Qdrant.
    """
    raw = f"{document_id}:{chunk_index}:{content_hash}"
    return str(uuid_lib.uuid5(POINT_ID_NAMESPACE, raw))


def generate_content_hash(content: str) -> str:
    """Generate a SHA256 hash of chunk text content."""
    return hashlib.sha256(content.encode()).hexdigest()
