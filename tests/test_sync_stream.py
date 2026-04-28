import pytest

from app.services.sync_stream import async_wrap_sync_generator


def _fake_sync_gen(chunks: list[str]):
    """Simulates a sync generator like ReactAgent.execute_stream."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_async_wrap_yields_all_chunks():
    async for _ in async_wrap_sync_generator(_fake_sync_gen([])):
        pass  # empty generator should not yield anything

    collected = []
    async for chunk in async_wrap_sync_generator(_fake_sync_gen(["a", "b", "c"])):
        collected.append(chunk)
    assert collected == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_async_wrap_empty_generator():
    collected = []
    async for chunk in async_wrap_sync_generator(_fake_sync_gen([])):
        collected.append(chunk)
    assert collected == []


@pytest.mark.asyncio
async def test_async_wrap_single_chunk():
    collected = []
    async for chunk in async_wrap_sync_generator(_fake_sync_gen(["only"])):
        collected.append(chunk)
    assert collected == ["only"]
