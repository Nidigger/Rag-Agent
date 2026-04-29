"""Sync-to-async generator bridge.

Wraps synchronous generators (e.g. FinalAnswerStreamer.stream_final_answer)
so they can be consumed by async FastAPI SSE endpoints without blocking
the event loop. Uses a ThreadPoolExecutor for non-blocking iteration.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

_executor = ThreadPoolExecutor(max_workers=4)
_SENTINEL = object()

logger = logging.getLogger("rag-agent.sync_stream")


def _next_or_sentinel(gen: Generator) -> object:
    """Advance the generator, returning _SENTINEL on StopIteration."""
    try:
        return next(gen)
    except StopIteration:
        return _SENTINEL


async def async_wrap_sync_generator(gen: Generator):
    """Wrap a synchronous generator into an async generator.

    Each iteration runs in a thread pool to avoid blocking the
    asyncio event loop. Useful for bridging sync LangChain generators
    to async FastAPI SSE handlers.
    """
    loop = asyncio.get_event_loop()
    while True:
        result = await loop.run_in_executor(_executor, _next_or_sentinel, gen)
        if result is _SENTINEL:
            break
        yield result
