"""Stream event factory functions and tool status message mapping.

Centralizes the construction of SSE event dicts so that service and API
layers never hand-craft event structures.
"""

from __future__ import annotations

from app.schemas.stream import StreamEvent, StreamPhase

# ---------------------------------------------------------------------------
# Tool status message mapping — user-friendly Chinese text per tool
# ---------------------------------------------------------------------------

TOOL_STATUS_MESSAGES: dict[str, dict[str, str]] = {
    "rag_summarize": {
        "start": "正在检索相关资料",
        "done": "相关资料已检索完成",
        "error": "资料检索失败",
    },
    "fetch_external_data": {
        "start": "正在获取用户使用数据",
        "done": "用户使用数据已获取",
        "error": "用户使用数据获取失败",
    },
    "get_weather": {
        "start": "正在获取天气信息",
        "done": "天气信息已获取",
        "error": "天气信息获取失败",
    },
    "get_user_location": {
        "start": "正在确认用户所在城市",
        "done": "用户所在城市已确认",
        "error": "用户城市确认失败",
    },
    "get_user_id": {
        "start": "正在确认用户身份",
        "done": "用户身份已确认",
        "error": "用户身份确认失败",
    },
    "get_current_month": {
        "start": "正在确认报告月份",
        "done": "报告月份已确认",
        "error": "报告月份确认失败",
    },
    "fill_context_for_report": {
        "start": "正在准备报告上下文",
        "done": "报告上下文已准备完成",
        "error": "报告上下文准备失败",
    },
}

_FALLBACK_MESSAGES: dict[str, str] = {
    "start": "正在调用工具",
    "done": "工具调用完成",
    "error": "工具调用失败",
}


def get_tool_status_message(tool_name: str, status: str) -> str:
    """Return a user-friendly message for a tool status transition."""
    return TOOL_STATUS_MESSAGES.get(tool_name, _FALLBACK_MESSAGES).get(
        status, _FALLBACK_MESSAGES[status]
    )


# ---------------------------------------------------------------------------
# Event factory functions
# ---------------------------------------------------------------------------


def status_event(phase: StreamPhase, message: str) -> StreamEvent:
    """Create a *status* event indicating the current streaming phase."""
    return {
        "event": "status",
        "data": {"phase": phase, "message": message},
    }


def message_event(content: str) -> StreamEvent:
    """Create a *message* event carrying a content token."""
    return {
        "event": "message",
        "data": {"content": content},
    }


def tool_start_event(tool: str, message: str) -> StreamEvent:
    """Create a *tool_start* event indicating a specific tool has begun."""
    return {
        "event": "tool_start",
        "data": {"tool": tool, "message": message},
    }


def tool_done_event(tool: str, message: str) -> StreamEvent:
    """Create a *tool_done* event indicating a specific tool has finished."""
    return {
        "event": "tool_done",
        "data": {"tool": tool, "message": message},
    }


def done_event(session_id: str) -> StreamEvent:
    """Create a *done* event carrying the session id."""
    return {
        "event": "done",
        "data": {"session_id": session_id},
    }


def error_event(code: str, message: str) -> StreamEvent:
    """Create an *error* event with an error code and message."""
    return {
        "event": "error",
        "data": {"code": code, "message": message},
    }
