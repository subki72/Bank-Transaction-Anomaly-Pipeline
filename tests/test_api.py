"""Integration tests for FastAPI endpoints, schema validation, rate limiting, and error boundary."""

from __future__ import annotations

from fastapi.testclient import TestClient

from bank_anomaly.api import create_app
from bank_anomaly.config import PipelineSettings


def _sample_transaction(
    tx_id: str = "TX900001",
    amount: float = 145.0,
    attempts: float = 1.0,
    balance: float = 4200.0,
) -> dict[str, object]:
    """Construct a valid API transaction payload."""
    return {
        "TransactionID": tx_id,
        "AccountID": "AC00123",
        "TransactionAmount": amount,
        "PreviousTransactionDate": "2024-09-01 12:00:00",
        "TransactionType": "Debit",
        "Location": "Austin",
        "DeviceID": "D000101",
        "IP Address": "192.0.2.44",
        "MerchantID": "M012",
        "Channel": "Online",
        "CustomerAge": 34.0,
        "CustomerOccupation": "Engineer",
        "TransactionDuration": 95.0,
        "LoginAttempts": attempts,
        "AccountBalance": balance,
        "TransactionDate": "2024-11-04 08:15:00",
    }


def test_health_and_model_info_endpoints() -> None:
    """Health and model-info endpoints report operational readiness."""
    app = create_app(PipelineSettings(environment="test"))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "healthy"
        assert body["model_ready"] is True
        assert body["n_training_rows"] == 400

        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["model_name"] == "random_forest"
        assert info.json()["n_clusters"] == 2


def test_analyze_endpoint_scores_normal_and_anomalous_batch() -> None:
    """POST /analyze validates payloads, scores anomalies, and populates summary metrics."""
    app = create_app(PipelineSettings(environment="test"))
    payload = {
        "include_classification": True,
        "transactions": [
            _sample_transaction("TX_NORM", amount=85.0, attempts=1.0, balance=4800.0),
            _sample_transaction("TX_ANOM", amount=6500.0, attempts=6.0, balance=120.0),
        ],
    }

    with TestClient(app) as client:
        resp = client.post("/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_transactions"] == 2
        assert len(data["results"]) == 2
        norm_res, anom_res = data["results"]
        assert norm_res["transaction_id"] == "TX_NORM"
        assert anom_res["transaction_id"] == "TX_ANOM"
        assert anom_res["anomaly_score"] > norm_res["anomaly_score"]
        assert anom_res["is_anomaly"] is True
        assert anom_res["suspicious_probability"] is not None

        # Second identical request exercises TTL prediction cache cleanly
        cached_resp = client.post("/analyze", json=payload)
        assert cached_resp.status_code == 200
        assert cached_resp.json()["results"] == data["results"]


def test_analyze_validation_error_and_rate_limiting() -> None:
    """API rejects invalid schemas with 422 and enforces rate limits with 429."""
    app = create_app(PipelineSettings(environment="test", rate_limit_per_minute=3))
    with TestClient(app) as client:
        invalid = client.post("/analyze", json={"transactions": []})
        assert invalid.status_code == 422

        valid_req = {"include_classification": False, "transactions": [_sample_transaction()]}
        assert client.post("/analyze", json=valid_req).status_code == 200
        assert client.post("/analyze", json=valid_req).status_code == 200
        limited = client.post("/analyze", json=valid_req)
        assert limited.status_code == 429
        assert "Rate limit" in limited.json()["detail"]


def test_global_exception_handler_masks_internal_tracebacks() -> None:
    """Global exception handler returns safe 500 JSON without leaking internal stack details."""
    app = create_app(PipelineSettings(environment="test"))

    @app.get("/_boom")
    def trigger_boom() -> None:
        raise RuntimeError("sensitive internal traceback message")

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/_boom", headers={"X-Request-ID": "req-test-123"})
        assert resp.status_code == 500
        body = resp.json()
        assert body == {"detail": "Internal server error", "request_id": "req-test-123"}
