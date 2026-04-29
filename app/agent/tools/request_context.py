"""Per-request context that agent tools can read.

Set by the monitor_tool middleware before each tool execution,
and cleared after the tool returns.

Uses thread-local storage keyed by thread identity, which works
correctly because the agent runs tools synchronously within a
single thread (via ThreadPoolExecutor in sync_stream.py).
"""

import threading

_thread_local = threading.local()


def set_request_context(ctx: dict) -> None:
    """Store the current request context in thread-local storage."""
    _thread_local.ctx = ctx


def get_request_context() -> dict:
    """Retrieve the current request context (empty dict if not set)."""
    return getattr(_thread_local, "ctx", {})


def clear_request_context() -> None:
    """Clear the request context after tool execution."""
    _thread_local.ctx = {}
