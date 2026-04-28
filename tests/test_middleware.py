import pytest


@pytest.mark.asyncio
async def test_request_id_header(client):
    resp = await client.get("/nonexistent")
    assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_custom_request_id_forwarded(client):
    resp = await client.get(
        "/nonexistent", headers={"X-Request-ID": "test-req-123"}
    )
    assert resp.headers["X-Request-ID"] == "test-req-123"


@pytest.mark.asyncio
async def test_timing_header(client):
    resp = await client.get("/nonexistent")
    assert "X-Process-Time" in resp.headers
    assert float(resp.headers["X-Process-Time"]) >= 0


@pytest.mark.asyncio
async def test_not_found_returns_error(client):
    resp = await client.get("/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert "code" in body
    assert "error_code" in body
    assert "message" in body


@pytest.mark.asyncio
async def test_cors_headers(client):
    resp = await client.options(
        "/nonexistent",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" in resp.headers
