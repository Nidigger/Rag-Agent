"""Tests for ChatService and ReportService two-phase architecture.

Validates that:
- ChatService calls agent.execute() then streamer.stream_final_answer().
- ReportService passes report=True to the streamer.
- Both services yield the streamed chunks via async_wrap_sync_generator.
- Errors from the agent phase are wrapped in AgentGenerationError.
"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

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


def _make_fake_streamer(chunks):
    """Create a fake streamer whose stream_final_answer yields given chunks."""

    def fake_stream(query, tool_context, final_draft=None, report=False):
        for c in chunks:
            yield {"event": "message", "data": {"content": c}}

    streamer = MagicMock()
    streamer.stream_final_answer = fake_stream
    return streamer


# ---------------------------------------------------------------------------
# ChatService tests
# ---------------------------------------------------------------------------

class TestChatServiceTwoPhase:
    @pytest.mark.asyncio
    async def test_chat_service_yields_streamed_events(self):
        from app.services.chat_service import ChatService

        fake_result = _make_agent_result()
        fake_streamer = _make_fake_streamer(["chunk1", "chunk2"])

        with patch.object(
            ChatService, "__init__", lambda self: None
        ):
            svc = ChatService()
            svc._agent = MagicMock()
            svc._agent.execute.return_value = fake_result
            svc._streamer = fake_streamer

            collected = []
            async for event in svc.stream_chat("hello"):
                collected.append(event)

            assert len(collected) >= 2
            event_types = [e["event"] for e in collected]
            assert "status" in event_types
            assert "message" in event_types

    @pytest.mark.asyncio
    async def test_chat_service_calls_agent_with_report_false_and_event_sink(self):
        from app.services.chat_service import ChatService

        fake_result = _make_agent_result()
        fake_streamer = _make_fake_streamer(["ok"])

        with patch.object(
            ChatService, "__init__", lambda self: None
        ):
            svc = ChatService()
            svc._agent = MagicMock()
            svc._agent.execute.return_value = fake_result
            svc._streamer = fake_streamer

            async for _ in svc.stream_chat("hello"):
                pass

        svc._agent.execute.assert_called_once()
        call_kwargs = svc._agent.execute.call_args
        ctx = call_kwargs.kwargs.get("context", {})
        assert ctx["report"] is False
        assert "event_sink" in ctx

    @pytest.mark.asyncio
    async def test_chat_service_passes_messages_to_agent_and_event_sink(self):
        from app.services.chat_service import ChatService

        fake_result = _make_agent_result()
        fake_streamer = _make_fake_streamer(["ok"])
        history = [{"role": "user", "content": "prev"}]

        with patch.object(
            ChatService, "__init__", lambda self: None
        ):
            svc = ChatService()
            svc._agent = MagicMock()
            svc._agent.execute.return_value = fake_result
            svc._streamer = fake_streamer

            async for _ in svc.stream_chat("hello", messages=history):
                pass

        svc._agent.execute.assert_called_once()
        call_kwargs = svc._agent.execute.call_args
        assert call_kwargs.kwargs.get("messages") == history
        ctx = call_kwargs.kwargs.get("context", {})
        assert ctx["report"] is False
        assert "event_sink" in ctx

    @pytest.mark.asyncio
    async def test_chat_service_wraps_generic_error(self):
        from app.services.chat_service import ChatService

        with patch.object(
            ChatService, "__init__", lambda self: None
        ):
            svc = ChatService()
            svc._agent = MagicMock()
            svc._agent.execute.side_effect = RuntimeError("boom")
            svc._streamer = _make_fake_streamer([])

            with pytest.raises(AgentGenerationError):
                async for _ in svc.stream_chat("hello"):
                    pass

    @pytest.mark.asyncio
    async def test_chat_service_reraises_agent_error(self):
        from app.services.chat_service import ChatService

        with patch.object(
            ChatService, "__init__", lambda self: None
        ):
            svc = ChatService()
            svc._agent = MagicMock()
            svc._agent.execute.side_effect = AgentGenerationError("agent fail")
            svc._streamer = _make_fake_streamer([])

            with pytest.raises(AgentGenerationError, match="agent fail"):
                async for _ in svc.stream_chat("hello"):
                    pass


# ---------------------------------------------------------------------------
# ReportService tests
# ---------------------------------------------------------------------------

class TestReportServiceTwoPhase:
    @pytest.mark.asyncio
    async def test_report_service_yields_streamed_events(self):
        from app.services.report_service import ReportService

        fake_result = _make_agent_result(
            draft="report draft", context="external data"
        )

        captured_report = None

        def fake_stream(query, tool_context, final_draft=None, report=False):
            nonlocal captured_report
            captured_report = report
            yield {"event": "message", "data": {"content": "report_chunk"}}

        with patch.object(
            ReportService, "__init__", lambda self: None
        ):
            svc = ReportService()
            svc._agent = MagicMock()
            svc._agent.execute.return_value = fake_result
            svc._streamer = MagicMock()
            svc._streamer.stream_final_answer = fake_stream

            collected = []
            async for event in svc.stream_report(
                "generate report", user_id="1001", month="2025-06"
            ):
                collected.append(event)

        assert len(collected) >= 2
        event_types = [e["event"] for e in collected]
        assert "status" in event_types
        assert "message" in event_types
        assert captured_report is True

    @pytest.mark.asyncio
    async def test_report_service_passes_context_to_agent(self):
        from app.services.report_service import ReportService

        fake_result = _make_agent_result()
        fake_streamer = _make_fake_streamer(["ok"])

        with patch.object(
            ReportService, "__init__", lambda self: None
        ):
            svc = ReportService()
            svc._agent = MagicMock()
            svc._agent.execute.return_value = fake_result
            svc._streamer = fake_streamer

            async for _ in svc.stream_report(
                "report", user_id="1001", month="2025-06", device_id="dev1"
            ):
                pass

        svc._agent.execute.assert_called_once()
        call_kwargs = svc._agent.execute.call_args
        ctx = call_kwargs.kwargs.get("context", {})
        assert ctx["report"] is True
        assert ctx["user_id"] == "1001"
        assert ctx["month"] == "2025-06"
        assert ctx["device_id"] == "dev1"
        assert "event_sink" in ctx

    @pytest.mark.asyncio
    async def test_report_service_context_without_optional_fields(self):
        from app.services.report_service import ReportService

        fake_result = _make_agent_result()
        fake_streamer = _make_fake_streamer(["ok"])

        with patch.object(
            ReportService, "__init__", lambda self: None
        ):
            svc = ReportService()
            svc._agent = MagicMock()
            svc._agent.execute.return_value = fake_result
            svc._streamer = fake_streamer

            async for _ in svc.stream_report("report"):
                pass

        call_kwargs = svc._agent.execute.call_args
        ctx = call_kwargs.kwargs.get("context", {})
        assert ctx["report"] is True
        assert "event_sink" in ctx
