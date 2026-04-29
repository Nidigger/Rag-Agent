"""Concurrency regression tests — verify Phase 1 Agent execution does not block event loop.

These tests prove that:
- Slow Agent execution runs in a thread pool via asyncio.to_thread().
- The event loop remains responsive during Agent execution.
- Concurrent requests are not blocked by slow Agent phases.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.common.exceptions import AgentGenerationError
from app.schemas.common import AgentExecutionResult


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

def _make_agent_result(
    draft="final draft",
    context="tool context",
    tools=None,
):
    return AgentExecutionResult(
        final_draft=draft,
        tool_context=context,
        used_tools=tools or ["rag_summarize"],
        messages=[],
    )


class SlowAgent:
    """Agent that blocks for a given number of seconds via time.sleep()."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.execute_count = 0

    def execute(self, *args, **kwargs):
        self.execute_count += 1
        time.sleep(self.delay)
        return _make_agent_result()


class FakeStreamer:
    """Streamer that yields a list of chunks synchronously."""

    def __init__(self, chunks):
        self.chunks = chunks

    def stream_final_answer(self, query, tool_context, final_draft=None, report=False):
        for c in self.chunks:
            yield c


# ---------------------------------------------------------------------------
# Event loop non-blocking tests
# ---------------------------------------------------------------------------

class TestAgentDoesNotBlockEventLoop:
    """Verify that asyncio.to_thread prevents Agent from blocking event loop."""

    @pytest.mark.asyncio
    async def test_chat_service_agent_phase_non_blocking(self):
        """ChatService: slow Agent should not block asyncio.sleep."""
        from app.services.chat_service import ChatService

        with patch.object(ChatService, "__init__", lambda self: None):
            svc = ChatService()
            svc._agent = SlowAgent(delay=0.5)
            svc._streamer = FakeStreamer(["ok"])

            task = asyncio.create_task(_consume(svc.stream_chat("hello")))

            start = time.perf_counter()
            await asyncio.sleep(0.1)
            elapsed = time.perf_counter() - start

            assert elapsed < 0.4, (
                "Event loop was blocked during Agent execution. "
                "Agent should run in a thread pool."
            )
            await task

    @pytest.mark.asyncio
    async def test_report_service_agent_phase_non_blocking(self):
        """ReportService: slow Agent should not block asyncio.sleep."""
        from app.services.report_service import ReportService

        with patch.object(ReportService, "__init__", lambda self: None):
            svc = ReportService()
            svc._agent = SlowAgent(delay=0.5)
            svc._streamer = FakeStreamer(["report_ok"])

            task = asyncio.create_task(_consume(svc.stream_report("generate report")))

            start = time.perf_counter()
            await asyncio.sleep(0.1)
            elapsed = time.perf_counter() - start

            assert elapsed < 0.4, (
                "Event loop was blocked during Agent execution. "
                "Agent should run in a thread pool."
            )
            await task


class TestConcurrentRequests:
    """Verify that concurrent requests are handled in parallel."""

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests_are_parallel(self):
        """Two concurrent chat requests should run in parallel, not sequentially."""
        from app.services.chat_service import ChatService

        def make_service():
            with patch.object(ChatService, "__init__", lambda self: None):
                svc = ChatService()
                svc._agent = SlowAgent(delay=0.3)
                svc._streamer = FakeStreamer(["ok"])
                return svc

        svc1 = make_service()
        svc2 = make_service()

        start = time.perf_counter()

        task1 = asyncio.create_task(_consume(svc1.stream_chat("hello1")))
        task2 = asyncio.create_task(_consume(svc2.stream_chat("hello2")))

        await asyncio.gather(task1, task2)

        elapsed = time.perf_counter() - start

        # If requests were sequential, elapsed would be ~0.6s.
        # With threading, they should overlap and complete in ~0.3-0.4s.
        assert elapsed < 0.55, (
            f"Concurrent requests took {elapsed:.2f}s, expected < 0.55s. "
            "Requests may be running sequentially instead of in parallel."
        )

    @pytest.mark.asyncio
    async def test_chat_and_report_run_concurrently(self):
        """A chat and a report request should not block each other."""
        from app.services.chat_service import ChatService
        from app.services.report_service import ReportService

        with patch.object(ChatService, "__init__", lambda self: None):
            chat_svc = ChatService()
            chat_svc._agent = SlowAgent(delay=0.3)
            chat_svc._streamer = FakeStreamer(["chat_ok"])

        with patch.object(ReportService, "__init__", lambda self: None):
            report_svc = ReportService()
            report_svc._agent = SlowAgent(delay=0.3)
            report_svc._streamer = FakeStreamer(["report_ok"])

        start = time.perf_counter()

        task1 = asyncio.create_task(_consume(chat_svc.stream_chat("hello")))
        task2 = asyncio.create_task(_consume(report_svc.stream_report("report")))

        await asyncio.gather(task1, task2)

        elapsed = time.perf_counter() - start

        assert elapsed < 0.55, (
            f"Chat+Report took {elapsed:.2f}s, expected < 0.55s. "
            "Requests may be blocking each other."
        )


class TestErrorHandlingConcurrency:
    """Verify error handling still works with threaded Agent execution."""

    class RaisingAgent:
        def execute(self, *args, **kwargs):
            raise RuntimeError("threaded agent error")

    @pytest.mark.asyncio
    async def test_chat_service_wraps_threaded_error(self):
        from app.services.chat_service import ChatService

        with patch.object(ChatService, "__init__", lambda self: None):
            svc = ChatService()
            svc._agent = self.RaisingAgent()
            svc._streamer = FakeStreamer([])

        with pytest.raises(AgentGenerationError):
            async for _ in svc.stream_chat("hello"):
                pass

    @pytest.mark.asyncio
    async def test_report_service_wraps_threaded_error(self):
        from app.services.report_service import ReportService

        with patch.object(ReportService, "__init__", lambda self: None):
            svc = ReportService()
            svc._agent = self.RaisingAgent()
            svc._streamer = FakeStreamer([])

        with pytest.raises(AgentGenerationError):
            async for _ in svc.stream_report("report"):
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _consume(async_gen):
    """Consume an async generator to completion."""
    async for _ in async_gen:
        pass
