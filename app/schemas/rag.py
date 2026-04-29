from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    request_id: str | None = Field(None, description="Request ID")
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of results")
    knowledge_base_id: str = Field("kb_default", description="Knowledge base filter")
