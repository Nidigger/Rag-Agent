import json
from unittest.mock import patch

import pytest


class MockChatService:
    def __init__(self, events=None):
        self._events = events or [
            {"event": "status", "data": {"phase": "thinking", "message": "正在理解你的问题"}},
            {"event": "status", "data": {"phase": "generating", "message": "正在生成最终回答"}},
            {"event": "message", "data": {"content": "Hello "}},
            {"event": "message", "data": {"content": "World"}},
        ]

    async def stream_chat(self, query, session_id=None, messages=None):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_chat_stream_returns_event_stream(client):
    mock_service = MockChatService()
    with patch(
        "app.api.v1.chat.get_chat_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/chat/stream",
            json={"message": "你好"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_chat_stream_contains_content_and_done(client):
    mock_service = MockChatService(events=[
        {"event": "status", "data": {"phase": "thinking", "message": "正在理解你的问题"}},
        {"event": "message", "data": {"content": "chunk1"}},
    ])
    with patch(
        "app.api.v1.chat.get_chat_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/chat/stream",
            json={"message": "你好"},
        )
        body = resp.text
        assert "chunk1" in body
        assert "session_id" in body


@pytest.mark.asyncio
async def test_chat_stream_empty_message_rejected(client):
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": ""},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_non_stream(client):
    mock_service = MockChatService(events=[
        {"event": "status", "data": {"phase": "thinking", "message": "正在理解你的问题"}},
        {"event": "message", "data": {"content": "Hello "}},
    ])
    with patch(
        "app.api.v1.chat.get_chat_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/chat",
            json={"message": "你好"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["session_id"]
        assert "Hello" in body["data"]["reply"]


@pytest.mark.asyncio
async def test_chat_stream_error_event(client):
    mock_service = MockChatService()

    async def error_gen(query, session_id=None, messages=None):
        raise RuntimeError("model failed")
        yield

    mock_service.stream_chat = error_gen

    with patch(
        "app.api.v1.chat.get_chat_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/chat/stream",
            json={"message": "你好"},
        )
        body = resp.text
        assert "error" in body
        assert "INTERNAL_SERVER_ERROR" in body
