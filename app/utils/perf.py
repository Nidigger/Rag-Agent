"""Performance timing and logging utilities.

Provides low-intrusiveness helpers for adding elapsed-time measurements
across the Agent execution chain, RAG retrieval, embedding, Qdrant
search, and streaming phases.

Usage:

    from app.utils.perf import log_perf, now_ms, elapsed_ms

    start = now_ms()
    result = do_work()
    log_perf("my_module", "work_done",
        request_id=request_id,
        elapsed_ms=elapsed_ms(start),
        result_count=len(result),
    )

Field naming convention (see docs/日志字段统一与关键链路埋点小改造需求文档.md):
    request_id, tenant_id, user_id, endpoint, duration_ms, model_name,
    tool_name, status, error_code, error_message, result_count, top_k,
    collection, batch_size

Legacy field aliases are automatically normalized:
    elapsed_ms -> duration_ms, tool -> tool_name,
    model -> model_name, error -> error_message
"""

import logging

from time import perf_counter

PERF_LOGGER = logging.getLogger("rag-agent.perf")

# Legacy field aliases — callers may still use old names; they are
# normalized to the canonical name before output.
FIELD_ALIASES = {
    "elapsed_ms": "duration_ms",
    "tool": "tool_name",
    "model": "model_name",
    "error": "error_message",
}

# Fields that appear first in the log line, in this exact order, so that
# grep / tail can easily spot the most important context.  Fields not
# present in the call are simply skipped.
PRIORITY_FIELDS = [
    "request_id",
    "tenant_id",
    "user_id",
    "endpoint",
    "status",
    "duration_ms",
]

# Maximum length for error_message to prevent log bloat.
_ERROR_MESSAGE_MAX_LEN = 300


def now_ms() -> float:
    """Return current monotonic time in milliseconds.

    Designed to be paired with ``elapsed_ms()``:

        start = now_ms()
        ...
        elapsed_ms(start)  # -> int
    """
    return perf_counter() * 1000


def elapsed_ms(start_ms: float) -> int:
    """Return the integer number of milliseconds since ``start_ms``."""
    return int(now_ms() - start_ms)


def log_perf(module: str, event: str, **fields) -> None:
    """Emit a structured performance log entry.

    Format::

        [perf][module] event key=value key=value

    Example::

        log_perf("chat_api", "sse_start",
                 request_id="req-xxx", session_id="sess-yyy",
                 message_len=8)

    Produces::

        [perf][chat_api] sse_start request_id=req-xxx session_id=sess-yyy message_len=8

    ``fields`` may include ``error`` for error-path entries. All values
    are converted via ``str()`` so that non-string types (int, float,
    bool) are accepted.

    Legacy field names (``elapsed_ms``, ``tool``, ``model``, ``error``)
    are automatically mapped to the canonical names
    (``duration_ms``, ``tool_name``, ``model_name``, ``error_message``).
    ``error_message`` values are truncated to 300 characters.
    """
    # 1. Normalize field names via aliases
    normalized: dict[str, object] = {}
    for key, val in fields.items():
        key = FIELD_ALIASES.get(key, key)
        if key == "error_message" and val is not None:
            val = str(val)[:_ERROR_MESSAGE_MAX_LEN]
        normalized[key] = val

    # 2. Build output: priority fields first (in order), then remaining alphabetically
    parts = [f"[perf][{module}]", event]
    for key in PRIORITY_FIELDS:
        if key in normalized:
            val = normalized.pop(key)
            parts.append(f"{key}={val}" if val is not None else f"{key}=None")
    for key in sorted(normalized):
        val = normalized[key]
        parts.append(f"{key}={val}" if val is not None else f"{key}=None")

    PERF_LOGGER.info(" ".join(parts))
