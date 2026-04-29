"""Knowledge base schemas — ingest and document management requests/responses."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request to ingest a document into the vector store."""

    document_id: str = Field(..., description="Stable document identifier")
    document_version_id: str | None = Field(None, description="Version identifier")
    file_path: str = Field(..., description="Absolute path to the document file")
    file_hash: str | None = Field(None, description="SHA256 hash of the file")
    knowledge_base_id: str = Field("kb_default", description="Knowledge base scope")
    tenant_id: str = Field("default", description="Tenant identifier")


class IngestResponse(BaseModel):
    """Response from a document ingest operation."""

    document_id: str
    chunk_count: int
    status: str


class DeleteDocumentRequest(BaseModel):
    """Request to delete document vectors."""

    document_id: str = Field(..., description="Document to delete")
    physical: bool = Field(False, description="True for physical delete, False for logical")


class DeleteDocumentResponse(BaseModel):
    """Response from a document deletion operation."""

    document_id: str
    status: str
