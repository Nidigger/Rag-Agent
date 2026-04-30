"""Tests for performance log correctness — model_done pairing, request_id propagation.

Validates that:
- emit_pending_model_done() emits a done log when _model_start exists.
- emit_pending_model_done() is a no-op when _model_start is absent.
- ReactAgent.execute() emits model_done even for no-tool queries.
- ReactAgent.execute() emits model_done for the final model round.
- set_perf_request_id() / get_perf_request_id() / clear_perf_request_id()
  work correctly for thread-local propagation.
"""

import time
from unittest.mock import patch

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.react_agent import ReactAgent
from app.agent.tools.middleware import emit_pending_model_done
from app.agent.tools.request_context import (
    clear_perf_request_id,
    get_perf_request_id,
    set_perf_request_id,
)
from app.utils.perf import now_ms


# ---------------------------------------------------------------------------
# Fake agent that simulates self._agent.invoke()
# ---------------------------------------------------------------------------

class _FakeAgentInvoke:
    def __init__(self, messages):
        self._messages = messages

    def invoke(self, input_dict, *, context=None):
        return {"messages": self._messages}


# ---------------------------------------------------------------------------
# emit_pending_model_done tests
# ---------------------------------------------------------------------------

class TestEmitPendingModelDone:
    @patch("app.agent.tools.middleware.log_perf")
    def test_emits_done_when_model_start_exists(self, mock_log_perf):
        ctx = {
            "_model_start": now_ms(),
            "_perf_step": 3,
            "request_id": "req-test-001",
        }
        emit_pending_model_done(ctx)

        mock_log_perf.assert_called_once()
        call_args = mock_log_perf.call_args
        assert call_args[0][0] == "agent_model"
        assert call_args[0][1] == "done"
        kwargs = call_args[1]
        assert kwargs["request_id"] == "req-test-001"
        assert kwargs["step"] == 3
        assert "elapsed_ms" in kwargs

    @patch("app.agent.tools.middleware.log_perf")
    def test_no_op_when_model_start_missing(self, mock_log_perf):
        ctx = {"request_id": "req-test-002"}
        emit_pending_model_done(ctx)

        mock_log_perf.assert_not_called()

    @patch("app.agent.tools.middleware.log_perf")
    def test_clears_model_start_after_emit(self, mock_log_perf):
        ctx = {
            "_model_start": now_ms(),
            "_perf_step": 1,
            "request_id": "req-test-003",
        }
        emit_pending_model_done(ctx)

        assert "_model_start" not in ctx
        assert "_perf_step" not in ctx
        assert "request_id" in ctx  # not cleared

    @patch("app.agent.tools.middleware.log_perf")
    def test_safe_to_call_multiple_times(self, mock_log_perf):
        ctx = {
            "_model_start": now_ms(),
            "request_id": "req-test-004",
        }
        emit_pending_model_done(ctx)
        emit_pending_model_done(ctx)  # second call should be no-op

        assert mock_log_perf.call_count == 1

    def test_fallback_request_id_internal(self):
        ctx = {"_model_start": now_ms()}
        emit_pending_model_done(ctx)
        # Should not raise — falls back to "internal"


# ---------------------------------------------------------------------------
# ReactAgent model_done pairing tests
# ---------------------------------------------------------------------------

class TestReactAgentModelDonePairing:
    @patch("app.agent.tools.middleware.log_perf")
    def test_no_tool_still_emits_model_done(self, mock_log_perf):
        """No-tool query should produce agent_model done via pending emit.

        The fake agent bypasses the LangChain middleware loop, so we
        pre-set _model_start in the context to simulate what
        log_before_model would do in a real agent run.
        """
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="Hi there!"),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("hello", context={
            "request_id": "req-mt1",
            "_model_start": now_ms(),
            "_perf_step": 1,
        })
        assert result.final_draft == "Hi there!"

        done_calls = [
            c for c in mock_log_perf.call_args_list
            if c[0][0] == "agent_model" and c[0][1] == "done"
        ]
        assert len(done_calls) >= 1

    @patch("app.agent.tools.middleware.log_perf")
    def test_tool_calls_emit_model_done_for_final_round(self, mock_log_perf):
        """The last model round (no tools) must have a done event."""
        messages = [
            HumanMessage(content="make report"),
            AIMessage(content="", tool_calls=[
                {"name": "fetch_external_data", "args": {"user_id": "1001", "month": "2025-06"}, "id": "c1"},
            ]),
            ToolMessage(content="raw external data", tool_call_id="c1", name="fetch_external_data"),
            AIMessage(content="Here is your report summary."),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("make report", context={
            "request_id": "req-mt2",
            "_model_start": now_ms(),
            "_perf_step": 2,
        })
        assert "report summary" in result.final_draft

        done_calls = [
            c for c in mock_log_perf.call_args_list
            if c[0][0] == "agent_model" and c[0][1] == "done"
        ]
        assert len(done_calls) >= 1


# ---------------------------------------------------------------------------
# request_id thread-local propagation tests
# ---------------------------------------------------------------------------

class TestPerfRequestIdPropagation:
    def test_set_and_get(self):
        set_perf_request_id("req-prop-001")
        assert get_perf_request_id() == "req-prop-001"

    def test_default_is_internal(self):
        clear_perf_request_id()
        assert get_perf_request_id() == "internal"

    def test_clear_removes(self):
        set_perf_request_id("req-prop-002")
        clear_perf_request_id()
        assert get_perf_request_id() == "internal"

    def test_overwrite(self):
        set_perf_request_id("req-prop-003")
        set_perf_request_id("req-prop-004")
        assert get_perf_request_id() == "req-prop-004"

    def test_tool_context_provides_request_id(self):
        """Simulate the monitor_tool flow: context → set_request_context + set_perf_request_id."""
        from app.agent.tools.request_context import (
            clear_request_context,
            get_request_context,
            set_request_context,
        )

        ctx = {
            "request_id": "req-from-context",
            "report": True,
        }
        set_request_context(ctx)
        set_perf_request_id(ctx.get("request_id", "internal"))

        # Both should read back correctly
        assert get_request_context().get("request_id") == "req-from-context"
        assert get_perf_request_id() == "req-from-context"

        # Cleanup as monitor_tool finally would
        clear_request_context()
        clear_perf_request_id()

        assert get_request_context().get("request_id", None) is None
        assert get_perf_request_id() == "internal"
