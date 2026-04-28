import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

_executor = ThreadPoolExecutor(max_workers=4)
_SENTINEL = object()

logger = logging.getLogger("rag-agent.sync_stream")


def _next_or_sentinel(gen: Generator) -> object:
    try:
        return next(gen)
    except StopIteration:
        return _SENTINEL


async def async_wrap_sync_generator(gen: Generator):
    """Wrap a synchronous generator into an async generator.

    Uses a thread pool to avoid blocking the event loop.
    Handles StopIteration correctly across Python versions.
    """
    loop = asyncio.get_event_loop()
    while True:
        result = await loop.run_in_executor(_executor, _next_or_sentinel, gen)
        if result is _SENTINEL:
            break
        yield result
