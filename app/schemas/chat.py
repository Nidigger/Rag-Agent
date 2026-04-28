from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    request_id: str | None = Field(None, description="Request ID")
    session_id: str | None = Field(None, description="Session ID")
    user_id: str | None = Field(None, description="User ID")
    device_id: str | None = Field(None, description="Device ID")
    message: str = Field(
        ..., min_length=1, max_length=2000, description="User message"
    )
    context: dict | None = Field(
        None, description="Additional context (city, month, etc.)"
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: dict | None = None


class SSEEvent(BaseModel):
    content: str | None = None
    done: bool | None = None
    session_id: str | None = None
    error: str | None = None
    message: str | None = None
