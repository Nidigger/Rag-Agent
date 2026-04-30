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


class AgentExecutionResult(BaseModel):
    """Result from the Agent tool-orchestration phase.

    Attributes:
        final_draft: The last AIMessage content produced by the Agent,
            intended as a draft for the streaming phase to refine.
        tool_context: Concatenated ToolMessage contents collected during
            execution, providing factual context for the final answer.
        used_tools: Unique names of tools invoked during this execution.
        tool_call_count: Total number of ToolMessage records, including
            repeated calls to the same tool.
        messages: Full message history from the Agent run, useful for
            debugging and auditing.
    """

    final_draft: str
    tool_context: str
    used_tools: list[str] = Field(default_factory=list)
    tool_call_count: int = 0
    messages: list = Field(default_factory=list)
