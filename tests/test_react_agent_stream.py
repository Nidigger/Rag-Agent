"""Tests for ReactAgent.execute() — the non-streaming tool orchestration phase.

Validates that execute():
- Returns an AgentExecutionResult with correct fields.
- Collects tool context from ToolMessage instances.
- Tracks used tool names.
- Extracts the final AIMessage (no tool_calls) as final_draft.
- Handles empty results, tool-call-only runs, and list-content messages.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.react_agent import ReactAgent, _message_content_to_text
from app.schemas.common import AgentExecutionResult


# ---------------------------------------------------------------------------
# Fake agent that simulates self._agent.invoke() return value
# ---------------------------------------------------------------------------

class _FakeAgentInvoke:
    """Simulates a LangChain agent's .invoke() method."""

    def __init__(self, messages):
        self._messages = messages

    def invoke(self, input_dict, *, context=None):
        return {"messages": self._messages}


def _make_tool_call_agent_messages():
    """Full agent loop: user → tool_call → tool_result → final answer."""
    return [
        HumanMessage(content="make report"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_external_data",
                    "args": {"user_id": "1001", "month": "2025-06"},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(
            content="raw external data",
            tool_call_id="call_1",
            name="fetch_external_data",
        ),
        AIMessage(content="Here is your report summary."),
    ]


def _make_no_tool_agent_messages():
    """Simple loop: user → direct answer (no tools)."""
    return [
        HumanMessage(content="hello"),
        AIMessage(content="Hi there!"),
    ]


def _make_tool_only_no_final_answer():
    """Edge case: tool calls but no final AIMessage without tool_calls."""
    return [
        HumanMessage(content="check data"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "深圳"},
                    "id": "call_2",
                }
            ],
        ),
        ToolMessage(
            content="晴天 26℃",
            tool_call_id="call_2",
            name="get_weather",
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReactAgentExecute:
    """Test ReactAgent.execute() with a fake agent backend."""

    def test_execute_returns_result_with_tool_calls(self):
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(_make_tool_call_agent_messages())

        result = agent.execute("make report", context={"report": True})

        assert isinstance(result, AgentExecutionResult)
        assert result.final_draft == "Here is your report summary."
        assert "raw external data" in result.tool_context
        assert "fetch_external_data" in result.used_tools
        assert len(result.messages) == 4

    def test_execute_no_tool_calls(self):
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(_make_no_tool_agent_messages())

        result = agent.execute("hello")

        assert result.final_draft == "Hi there!"
        assert result.tool_context == ""
        assert result.used_tools == []
        assert len(result.messages) == 2

    def test_execute_tool_calls_no_final_answer(self):
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(_make_tool_only_no_final_answer())

        result = agent.execute("check data")

        # No final AIMessage without tool_calls → empty draft
        assert result.final_draft == ""
        assert "晴天 26℃" in result.tool_context
        assert "get_weather" in result.used_tools

    def test_execute_default_context_is_report_false(self):
        """When context is None, it defaults to {"report": False}."""
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(_make_no_tool_agent_messages())

        result = agent.execute("hello")
        assert isinstance(result, AgentExecutionResult)

    def test_execute_truncates_long_history(self):
        """History longer than 10 messages should be truncated."""
        agent = ReactAgent()
        long_history = [{"role": "user", "content": f"msg{i}"} for i in range(15)]
        agent._agent = _FakeAgentInvoke(_make_no_tool_agent_messages())

        # Should not raise — just truncates internally
        result = agent.execute("hello", messages=long_history)
        assert isinstance(result, AgentExecutionResult)

    def test_execute_tool_context_is_newline_separated(self):
        """Multiple ToolMessages should be joined with newlines."""
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="", tool_calls=[
                {"name": "t1", "args": {}, "id": "c1"},
            ]),
            ToolMessage(content="result1", tool_call_id="c1", name="t1"),
            AIMessage(content="", tool_calls=[
                {"name": "t2", "args": {}, "id": "c2"},
            ]),
            ToolMessage(content="result2", tool_call_id="c2", name="t2"),
            AIMessage(content="done"),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("test")
        assert "result1" in result.tool_context
        assert "result2" in result.tool_context
        assert "\n" in result.tool_context

    def test_execute_deduplicates_tool_names(self):
        """Same tool called twice should appear once in used_tools."""
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"city": "a"}, "id": "c1"},
            ]),
            ToolMessage(content="sunny", tool_call_id="c1", name="get_weather"),
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"city": "b"}, "id": "c2"},
            ]),
            ToolMessage(content="rainy", tool_call_id="c2", name="get_weather"),
            AIMessage(content="Both cities checked."),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("test")
        assert result.used_tools.count("get_weather") == 1
        assert result.used_tools == ["get_weather"]

    def test_execute_tool_call_count_reflects_raw_calls(self):
        """Repeated tool calls should increment tool_call_count."""
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"city": "a"}, "id": "c1"},
            ]),
            ToolMessage(content="sunny", tool_call_id="c1", name="get_weather"),
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"city": "b"}, "id": "c2"},
            ]),
            ToolMessage(content="rainy", tool_call_id="c2", name="get_weather"),
            AIMessage(content="Both cities checked."),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("test")
        # Used tools deduplicated — only 1 unique name
        assert len(result.used_tools) == 1
        # Raw tool_call_count includes both invocations
        assert result.tool_call_count == 2

    def test_execute_tool_call_count_multiple_unique(self):
        """Three unique tools, each called once."""
        messages = [
            HumanMessage(content="test"),
            AIMessage(content="", tool_calls=[
                {"name": "rag_summarize", "args": {"query": "how to clean"}, "id": "c1"},
            ]),
            ToolMessage(content="clean tips", tool_call_id="c1", name="rag_summarize"),
            AIMessage(content="", tool_calls=[
                {"name": "get_weather", "args": {"city": "shenzhen"}, "id": "c2"},
            ]),
            ToolMessage(content="sunny", tool_call_id="c2", name="get_weather"),
            AIMessage(content="", tool_calls=[
                {"name": "get_user_location", "args": {}, "id": "c3"},
            ]),
            ToolMessage(content="shenzhen", tool_call_id="c3", name="get_user_location"),
            AIMessage(content="Here is your report."),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("test")
        assert len(result.used_tools) == 3
        assert result.tool_call_count == 3

    def test_execute_tool_call_count_repeated_rag(self):
        """rag_summarize called 4 times, get_weather once."""
        messages = [
            HumanMessage(content="test"),
        ]
        for i in range(4):
            messages.append(AIMessage(content="", tool_calls=[
                {"name": "rag_summarize", "args": {"query": f"q{i}"}, "id": f"c{i}"},
            ]))
            messages.append(ToolMessage(
                content=f"result{i}", tool_call_id=f"c{i}", name="rag_summarize"
            ))
        messages.append(AIMessage(content="", tool_calls=[
            {"name": "get_weather", "args": {"city": "sz"}, "id": "c_weather"},
        ]))
        messages.append(ToolMessage(
            content="sunny", tool_call_id="c_weather", name="get_weather"
        ))
        messages.append(AIMessage(content="Final answer."))

        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("test")
        # used_tools deduplicated: only 2 unique
        assert len(result.used_tools) == 2
        assert "rag_summarize" in result.used_tools
        assert "get_weather" in result.used_tools
        # tool_call_count = raw count
        assert result.tool_call_count == 5


class TestAgentHandlesToolMessageWithoutName:
    """Regression: ToolMessage.name can be None in some LangChain scenarios."""

    def test_execute_handles_tool_message_without_name(self):
        messages = [
            HumanMessage(content="test"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "rag_summarize", "args": {"query": "x"}, "id": "c1"}
                ],
            ),
            ToolMessage(content="result", tool_call_id="c1"),
            AIMessage(content="done"),
        ]
        agent = ReactAgent()
        agent._agent = _FakeAgentInvoke(messages)

        result = agent.execute("test", context={"request_id": "req-no-tool-name"})

        assert result.final_draft == "done"
        assert result.tool_context == "result"
        assert result.used_tools == ["unknown"]
        assert result.tool_call_count == 1


class TestMessageContentToText:
    """Test the _message_content_to_text helper."""

    def test_string_content(self):
        assert _message_content_to_text("hello") == "hello"

    def test_list_of_strings(self):
        assert _message_content_to_text(["hello", " world"]) == "hello world"

    def test_list_of_dicts_with_text(self):
        content = [{"text": "hello"}, {"text": " world"}]
        assert _message_content_to_text(content) == "hello world"

    def test_list_mixed(self):
        content = ["hello", {"text": " world"}, {"type": "image"}]
        assert _message_content_to_text(content) == "hello world"

    def test_empty_string(self):
        assert _message_content_to_text("") == ""

    def test_integer_fallback(self):
        assert _message_content_to_text(42) == "42"

    def test_none_text_in_dict(self):
        content = [{"text": None}, {"text": "ok"}]
        assert _message_content_to_text(content) == "ok"
