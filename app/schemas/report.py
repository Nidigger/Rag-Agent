from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    request_id: str | None = Field(None, description="Request ID")
    session_id: str | None = Field(None, description="Session ID")
    user_id: str | None = Field(None, description="User ID")
    device_id: str | None = Field(None, description="Device ID")
    month: str | None = Field(None, description="Report month (YYYY-MM)")
