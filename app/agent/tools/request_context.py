"""Per-request context that agent tools can read.

Set by the monitor_tool middleware before each tool execution,
and cleared after the tool returns.

Uses thread-local storage keyed by thread identity, which works
correctly because the agent runs tools synchronously within a
single thread (via ThreadPoolExecutor in sync_stream.py).

Also provides a lightweight ``request_id`` propagation mechanism
for performance logging across deep layers (embedding, Qdrant, RAG)
without passing ``request_id`` through every function signature.
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


# ---------------------------------------------------------------------------
# Performance context — lightweight request_id propagation
# ---------------------------------------------------------------------------


def set_perf_request_id(request_id: str) -> None:
    """Store ``request_id`` for performance logging in the current thread.

    Called at entry points (chat API, report API, RAG query) so that
    deep layers (embedding client, Qdrant store, retriever) can read it
    via ``get_perf_request_id()`` without explicit parameter threading.
    """
    _thread_local.perf_request_id = request_id


def get_perf_request_id() -> str:
    """Return the current thread's ``request_id`` or ``"internal"``."""
    return getattr(_thread_local, "perf_request_id", "internal")


def clear_perf_request_id() -> None:
    """Clear the performance ``request_id`` for the current thread."""
    if hasattr(_thread_local, "perf_request_id"):
        del _thread_local.perf_request_id
