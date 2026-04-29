"""Tests for the Report SSE API endpoint."""

from unittest.mock import patch

import pytest


class MockReportService:
    def __init__(self, chunks=None):
        self._chunks = chunks or ["Report chunk"]

    async def stream_report(
        self, query, session_id=None, user_id=None, month=None, device_id=None
    ):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_report_stream_returns_event_stream(client):
    mock_service = MockReportService(chunks=["Part1", " Part2"])
    with patch(
        "app.api.v1.report.get_report_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/report/stream",
            json={"user_id": "1001", "month": "2025-06"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_report_stream_contains_content_and_done(client):
    mock_service = MockReportService(chunks=["report_data"])
    with patch(
        "app.api.v1.report.get_report_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/report/stream",
            json={"user_id": "1001", "month": "2025-06"},
        )
        body = resp.text
        assert "report_data" in body
        assert "session_id" in body


@pytest.mark.asyncio
async def test_report_stream_no_required_fields(client):
    """Report endpoint should work with no fields provided."""
    mock_service = MockReportService(chunks=["通用报告"])
    with patch(
        "app.api.v1.report.get_report_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/report/stream",
            json={},
        )
        assert resp.status_code == 200
        assert "通用报告" in resp.text


@pytest.mark.asyncio
async def test_report_stream_error_event(client):
    mock_service = MockReportService(chunks=None)

    async def error_gen(query, session_id=None, **kwargs):
        raise RuntimeError("model failed")
        yield  # make it a generator

    mock_service.stream_report = error_gen

    with patch(
        "app.api.v1.report.get_report_service", return_value=mock_service
    ):
        resp = await client.post(
            "/api/v1/report/stream",
            json={"user_id": "1001", "month": "2025-06"},
        )
        body = resp.text
        assert "error" in body
