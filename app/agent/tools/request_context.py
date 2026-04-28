"""Per-request context that agent tools can read.

Set by middleware before tool execution, cleared after.
Uses a simple module-level dict keyed by thread identity.
"""

import threading

_thread_local = threading.local()


def set_request_context(ctx: dict) -> None:
    _thread_local.ctx = ctx


def get_request_context() -> dict:
    return getattr(_thread_local, "ctx", {})


def clear_request_context() -> None:
    _thread_local.ctx = {}
