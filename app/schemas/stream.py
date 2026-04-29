"""SSE stream event type definitions."""

from typing import Literal, TypedDict


StreamPhase = Literal["thinking", "tool_calling", "generating"]


class StreamEvent(TypedDict):
    """Unified SSE event structure emitted by services.

    Attributes:
        event: SSE event name (status, message, tool_start, tool_done, done, error).
        data: Event payload as a plain dict, serialized to JSON by the API layer.
    """

    event: str
    data: dict
