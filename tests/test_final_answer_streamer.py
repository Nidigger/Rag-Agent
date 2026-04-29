"""Tests for app/services/final_answer_streamer.py.

Validates that FinalAnswerStreamer:
- Calls the streaming model with correct message types (System + Human).
- Yields text tokens from the model response.
- Uses report prompt when report=True.
- Uses chat prompt when report=False.
- Handles empty model output gracefully.
- get_final_answer_streamer returns a singleton.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.final_answer_streamer import (
    FinalAnswerStreamer,
    get_final_answer_streamer,
)


def _make_fake_stream_model(chunks):
    """Create a mock model whose .stream() returns the given text chunks."""

    class _FakeChunk:
        def __init__(self, content):
            self.content = content

    def fake_stream(messages):
        for chunk_text in chunks:
            yield _FakeChunk(chunk_text)

    model = MagicMock()
    model.stream = fake_stream
    return model


class TestFinalAnswerStreamer:
    def test_yields_text_tokens(self):
        model = _make_fake_stream_model(["Hello", " ", "World"])
        streamer = FinalAnswerStreamer()
        streamer._model = model

        result = list(streamer.stream_final_answer(
            query="hi",
            tool_context="",
            final_draft="",
        ))
        assert result == ["Hello", " ", "World"]

    def test_includes_query_in_prompt(self):
        """Verify the user prompt contains the original query."""
        captured_messages = []

        model = MagicMock()

        def fake_stream(messages):
            captured_messages.extend(messages)
            yield MagicMock(content="ok")

        model.stream = fake_stream
        streamer = FinalAnswerStreamer()
        streamer._model = model

        list(streamer.stream_final_answer(
            query="我的问题",
            tool_context="ctx",
            final_draft="draft",
        ))

        # Should have SystemMessage + HumanMessage
        assert len(captured_messages) == 2
        human_msg = captured_messages[1]
        assert "我的问题" in human_msg.content
        assert "ctx" in human_msg.content
        assert "draft" in human_msg.content

    def test_chat_prompt_used_by_default(self):
        """report=False should use the chat system prompt."""
        captured_messages = []

        model = MagicMock()

        def fake_stream(messages):
            captured_messages.extend(messages)
            yield MagicMock(content="ok")

        model.stream = fake_stream
        streamer = FinalAnswerStreamer()
        streamer._model = model

        list(streamer.stream_final_answer(
            query="hi", tool_context="", report=False,
        ))

        system_msg = captured_messages[0]
        assert "简洁" in system_msg.content  # chat prompt mentions 简洁

    def test_report_prompt_used_when_report_true(self):
        """report=True should use the report system prompt."""
        captured_messages = []

        model = MagicMock()

        def fake_stream(messages):
            captured_messages.extend(messages)
            yield MagicMock(content="ok")

        model.stream = fake_stream
        streamer = FinalAnswerStreamer()
        streamer._model = model

        list(streamer.stream_final_answer(
            query="生成报告", tool_context="", report=True,
        ))

        system_msg = captured_messages[0]
        assert "报告" in system_msg.content

    def test_empty_model_output_yields_nothing(self):
        model = _make_fake_stream_model([])
        streamer = FinalAnswerStreamer()
        streamer._model = model

        result = list(streamer.stream_final_answer(
            query="hi", tool_context="",
        ))
        assert result == []

    def test_skips_empty_content_chunks(self):
        """Chunks with empty/None content should not be yielded."""

        class _Chunk:
            def __init__(self, content):
                self.content = content

        def fake_stream(messages):
            yield _Chunk("hello")
            yield _Chunk("")
            yield _Chunk(None)
            yield _Chunk("world")

        model = MagicMock()
        model.stream = fake_stream
        streamer = FinalAnswerStreamer()
        streamer._model = model

        result = list(streamer.stream_final_answer(
            query="hi", tool_context="",
        ))
        assert result == ["hello", "world"]

    def test_handles_none_draft(self):
        """final_draft=None should not cause errors."""
        model = _make_fake_stream_model(["ok"])
        streamer = FinalAnswerStreamer()
        streamer._model = model

        result = list(streamer.stream_final_answer(
            query="hi", tool_context="", final_draft=None,
        ))
        assert result == ["ok"]


class TestGetFinalAnswerStreamer:
    def test_returns_singleton(self):
        s1 = get_final_answer_streamer()
        s2 = get_final_answer_streamer()
        assert s1 is s2

    def test_returns_final_answer_streamer_instance(self):
        streamer = get_final_answer_streamer()
        assert isinstance(streamer, FinalAnswerStreamer)
