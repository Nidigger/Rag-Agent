"""AgentEventSink — thread-safe bridge from agent thread to SSE event loop.

The agent executes inside a thread pool (via asyncio.to_thread). Tool
middleware runs inside that same thread and needs a way to push events
back to the async SSE generator running on the event loop.

AgentEventSink solves this by wrapping an asyncio.Queue and using
loop.call_soon_threadsafe() for thread-safe insertion.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.stream import StreamEvent


class AgentEventSink:
    """Thread-safe event emitter that bridges agent tools → SSE stream.

    Usage::

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        sink = AgentEventSink(loop, queue)

        # Pass sink via agent context; middleware calls sink.emit(...)
        # The SSE generator drains the queue to yield events.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[StreamEvent | None],
    ) -> None:
        self._loop = loop
        self._queue = queue

    def emit(self, event: StreamEvent) -> None:
        """Emit an event from any thread. Thread-safe."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
