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
        used_tools: Names of tools invoked during this execution.
        messages: Full message history from the Agent run, useful for
            debugging and auditing.
    """

    final_draft: str
    tool_context: str
    used_tools: list[str]
    messages: list
