import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("JWT_SECRET", "test-secret")

# Ensure the api package is importable from tests/
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from api.main import app

ALGO_CLASSES = {"Aggressive", "IS", "Passive", "TWAP", "VWAP"}

VALID_PAYLOAD = {
    "notional_usd": 1_000_000,
    "market_volatility": 0.3,
    "spread_bps": 5,
    "time_of_day": 1,
    "venue_liquidity_score": 0.8,
}

VALID_CREDS = {"username": "admin", "password": "admin123"}


@pytest_asyncio.fixture(scope="module")
async def client():
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "loaded"


@pytest.mark.asyncio
async def test_predict_without_token_returns_401(client: AsyncClient):
    resp = await client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_predict_with_valid_token_returns_algo(client: AsyncClient):
    token_resp = await client.post("/auth/token", json=VALID_CREDS)
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    pred_resp = await client.post("/predict", json=VALID_PAYLOAD, headers=headers)
    assert pred_resp.status_code == 200
    body = pred_resp.json()
    assert "predicted_algo" in body
    assert body["predicted_algo"] in ALGO_CLASSES
    assert "confidence_scores" in body
    assert len(body["confidence_scores"]) == 5
    for cls in ALGO_CLASSES:
        assert cls in body["confidence_scores"]
