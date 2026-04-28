from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from api.main import app
from src.config import PREDICTION_AUDIT_LOG_PATH


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert "status" in payload
        assert "model_loaded" in payload


def test_predict_endpoint_and_audit_log():
    before_count = 0
    if PREDICTION_AUDIT_LOG_PATH.exists():
        before_count = len(PREDICTION_AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines())

    test_payload = {
        "amount": 1499.75,
        "user_id": "user_00033",
        "merchant_id": "merchant_0010",
        "timestamp": "2026-04-28T22:10:00",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=test_payload)
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert isinstance(data["fraud_probability"], float)
        assert isinstance(data["is_fraud"], bool)
        assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert "top_features" in data["explanation"]
        assert len(data["explanation"]["top_features"]) <= 3

    assert PREDICTION_AUDIT_LOG_PATH.exists()
    lines = PREDICTION_AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= before_count + 1
    last_record = json.loads(lines[-1])
    assert "transaction_id" in last_record
    assert "fraud_probability" in last_record
