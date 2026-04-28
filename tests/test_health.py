import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["status"] == "healthy"
    assert "version" in body["data"]


@pytest.mark.asyncio
async def test_health_has_request_id(client):
    resp = await client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_health_has_timing(client):
    resp = await client.get("/api/v1/health")
    assert "X-Process-Time" in resp.headers
