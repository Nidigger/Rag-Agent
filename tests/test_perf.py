"""Unit tests for app/utils/perf.py.

Validates that:
- now_ms() returns a positive float.
- elapsed_ms() computes the correct delta.
- log_perf() logs in the expected [perf][module] event key=value format.
- log_perf() handles various value types (int, str, bool, None).
- log_perf() normalizes legacy field aliases (elapsed_ms, tool, model, error).
- log_perf() outputs priority fields first.
- log_perf() omits fields not passed by the caller.
"""

import logging
from unittest.mock import patch

import pytest

from app.utils.perf import elapsed_ms, log_perf, now_ms


class TestNowMs:
    def test_returns_positive_float(self):
        t = now_ms()
        assert isinstance(t, float)
        assert t > 0


class TestElapsedMs:
    def test_returns_non_negative_int(self):
        start = now_ms()
        elapsed = elapsed_ms(start)
        assert isinstance(elapsed, int)
        assert elapsed >= 0

    def test_increasing_over_time(self):
        start = now_ms()
        e1 = elapsed_ms(start)
        e2 = elapsed_ms(start)
        assert e2 >= e1


class TestLogPerf:
    @patch("app.utils.perf.PERF_LOGGER")
    def test_logs_in_expected_format(self, mock_logger):
        log_perf("chat_api", "sse_start",
                 request_id="req-123",
                 session_id="sess-456",
                 message_len=8)

        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "[perf][chat_api]" in msg
        assert "sse_start" in msg
        assert "request_id=req-123" in msg
        assert "session_id=sess-456" in msg
        assert "message_len=8" in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_non_priority_fields_are_sorted(self, mock_logger):
        log_perf("test", "event", z="last", a="first")

        msg = mock_logger.info.call_args[0][0]
        a_pos = msg.index("a=first")
        z_pos = msg.index("z=last")
        assert a_pos < z_pos

    @patch("app.utils.perf.PERF_LOGGER")
    def test_handles_none_value(self, mock_logger):
        log_perf("test", "event", field=None)

        msg = mock_logger.info.call_args[0][0]
        assert "field=None" in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_handles_bool_values(self, mock_logger):
        log_perf("test", "event", flag=True)

        msg = mock_logger.info.call_args[0][0]
        assert "flag=True" in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_handles_int_values(self, mock_logger):
        log_perf("test", "event", count=42)

        msg = mock_logger.info.call_args[0][0]
        assert "count=42" in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_handles_empty_fields(self, mock_logger):
        log_perf("test", "event")

        msg = mock_logger.info.call_args[0][0]
        assert "[perf][test]" in msg
        assert msg.strip().endswith("event")


class TestLogPerfAliases:
    """Validate that legacy field names are normalized to canonical names."""

    @patch("app.utils.perf.PERF_LOGGER")
    def test_alias_elapsed_ms_to_duration_ms(self, mock_logger):
        log_perf("test", "done", elapsed_ms=100)

        msg = mock_logger.info.call_args[0][0]
        assert "duration_ms=100" in msg
        assert "elapsed_ms" not in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_alias_tool_to_tool_name(self, mock_logger):
        log_perf("test", "done", tool="rag_summarize")

        msg = mock_logger.info.call_args[0][0]
        assert "tool_name=rag_summarize" in msg
        assert "tool=" not in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_alias_model_to_model_name(self, mock_logger):
        log_perf("test", "done", model="gpt-4")

        msg = mock_logger.info.call_args[0][0]
        assert "model_name=gpt-4" in msg
        assert "model=" not in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_alias_error_to_error_message(self, mock_logger):
        log_perf("test", "error", error="something went wrong")

        msg = mock_logger.info.call_args[0][0]
        assert "error_message=something went wrong" in msg
        # Old name should not appear
        assert "error=something" not in msg

    @patch("app.utils.perf.PERF_LOGGER")
    def test_error_message_truncated_at_300_chars(self, mock_logger):
        long_error = "x" * 500
        log_perf("test", "error", error=long_error)

        msg = mock_logger.info.call_args[0][0]
        assert "error_message=" in msg
        # The value should be truncated to 300 chars
        error_val = msg.split("error_message=")[1]
        assert len(error_val) == 300

    @patch("app.utils.perf.PERF_LOGGER")
    def test_canonical_names_pass_through(self, mock_logger):
        """Using the canonical names directly should work without aliasing."""
        log_perf("test", "done",
                 duration_ms=50,
                 tool_name="search",
                 model_name="gpt-4",
                 error_message="ok")

        msg = mock_logger.info.call_args[0][0]
        assert "duration_ms=50" in msg
        assert "tool_name=search" in msg
        assert "model_name=gpt-4" in msg
        assert "error_message=ok" in msg


class TestLogPerfPriority:
    """Validate that priority fields appear before non-priority fields."""

    @patch("app.utils.perf.PERF_LOGGER")
    def test_priority_fields_come_first(self, mock_logger):
        log_perf("test", "done",
                 request_id="req-1",
                 status="success",
                 duration_ms=100,
                 batch_size=10)

        msg = mock_logger.info.call_args[0][0]
        # request_id and status and duration_ms should come before batch_size
        req_pos = msg.index("request_id=req-1")
        status_pos = msg.index("status=success")
        dur_pos = msg.index("duration_ms=100")
        batch_pos = msg.index("batch_size=10")
        assert req_pos < batch_pos
        assert status_pos < batch_pos
        assert dur_pos < batch_pos

    @patch("app.utils.perf.PERF_LOGGER")
    def test_missing_fields_not_output(self, mock_logger):
        log_perf("test", "done",
                 request_id="req-1",
                 batch_size=10)

        msg = mock_logger.info.call_args[0][0]
        assert "tenant_id=" not in msg
        assert "user_id=" not in msg
        assert "endpoint=" not in msg
