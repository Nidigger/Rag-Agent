from typing import Any

from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    request_id: str | None = Field(None, description="Request ID")
    session_id: str | None = Field(None, description="Session ID")
    user_id: str | None = Field(None, description="User ID")
    device_id: str | None = Field(None, description="Device ID")


class UnifiedResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: Any = None


class ErrorResponse(BaseModel):
    code: int
    error_code: str
    message: str
    request_id: str | None = None
