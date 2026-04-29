"""Tests for app/schemas/common.py — AgentExecutionResult."""

import pytest

from app.schemas.common import AgentExecutionResult


class TestAgentExecutionResult:
    def test_basic_construction(self):
        result = AgentExecutionResult(
            final_draft="answer",
            tool_context="ctx",
            used_tools=["t1"],
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result.final_draft == "answer"
        assert result.tool_context == "ctx"
        assert result.used_tools == ["t1"]
        assert len(result.messages) == 1

    def test_empty_fields(self):
        result = AgentExecutionResult(
            final_draft="",
            tool_context="",
            used_tools=[],
            messages=[],
        )
        assert result.final_draft == ""
        assert result.used_tools == []

    def test_multiple_tools(self):
        result = AgentExecutionResult(
            final_draft="",
            tool_context="c1\nc2",
            used_tools=["rag_summarize", "get_weather"],
            messages=[],
        )
        assert len(result.used_tools) == 2

    def test_model_dump(self):
        result = AgentExecutionResult(
            final_draft="test",
            tool_context="",
            used_tools=[],
            messages=[],
        )
        data = result.model_dump()
        assert "final_draft" in data
        assert "tool_context" in data
        assert "used_tools" in data
        assert "messages" in data
