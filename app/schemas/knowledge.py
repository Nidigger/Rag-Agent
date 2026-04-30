"""Knowledge base schemas — ingest and document management requests/responses."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StorageObject(BaseModel):
    """Reference to an object in a storage backend (e.g. MinIO)."""

    provider: Literal["minio"] = Field(
        "minio", description="Storage provider identifier"
    )
    bucket: str | None = Field(
        None,
        min_length=1,
        description="Bucket name; defaults to config",
    )
    object_key: str = Field(
        ...,
        min_length=1,
        description="Full object key within the bucket",
    )
    file_name: str | None = Field(
        None, description="Original filename for metadata"
    )


class IngestRequest(BaseModel):
    """Request to ingest a document into the vector store.

    Exactly one of ``file_path`` or ``storage`` must be provided.
    """

    document_id: str | None = Field(None, description="Stable document identifier")
    document_version_id: str | None = Field(None, description="Version identifier")
    file_path: str | None = Field(None, description="Absolute path to the document file")
    storage: StorageObject | None = Field(
        None, description="Object storage reference"
    )
    file_hash: str | None = Field(None, description="SHA256 hash of the file")
    knowledge_base_id: str = Field("kb_default", description="Knowledge base scope")
    tenant_id: str = Field("default", description="Tenant identifier")

    @model_validator(mode="after")
    def validate_source(self) -> "IngestRequest":
        """Ensure exactly one of file_path or storage is provided."""
        has_file_path = bool(self.file_path)
        has_storage = bool(self.storage)
        if has_file_path == has_storage:
            raise ValueError(
                "Exactly one of file_path or storage must be provided"
            )
        return self


class IngestedChunk(BaseModel):
    """Details of a single ingested chunk."""

    chunk_id: str = Field(..., description="Chunk identifier (document_id:version_id:chunk_index)")
    qdrant_collection: str = Field(..., description="Qdrant collection name")
    qdrant_point_id: str = Field(..., description="Qdrant point UUID")
    content_hash: str | None = None


class IngestResponse(BaseModel):
    """Response from a document ingest operation."""

    document_id: str
    chunk_count: int
    status: str
    chunks: list[IngestedChunk] | None = None


class DeleteDocumentRequest(BaseModel):
    """Request to delete document vectors."""

    document_id: str = Field(..., description="Document to delete")
    physical: bool = Field(False, description="True for physical delete, False for logical")


class DeleteDocumentResponse(BaseModel):
    """Response from a document deletion operation."""

    document_id: str
    status: str
