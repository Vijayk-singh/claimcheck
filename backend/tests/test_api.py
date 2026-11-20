"""
Tests for FastAPI Endpoints & Web Service.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify health check endpoint returns 200 OK and expected keys."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ClaimRisk AI API"


def test_predict_single_endpoint() -> None:
    """Verify POST /api/predict returns calibrated risk score and tier."""
    payload = {
        "claim_id": "CLM-API-01",
        "provider_id": "PRV-101",
        "procedure_code": "99214",
        "claim_amount": 250.0,
        "submission_delay": 14,
        "missing_fields": 0,
        "duplicate_indicator": 0,
        "provider_claim_frequency": 120,
        "historical_rejection_rate": 0.05,
        "amount_deviation": 0.1,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == "CLM-API-01"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}
    assert "explanation" in data


def test_predict_batch_endpoint() -> None:
    """Verify POST /api/predict/batch prioritizes claims queue."""
    payload = {
        "claims": [
            {
                "claim_id": "CLM-B1",
                "procedure_code": "99214",
                "claim_amount": 100.0,
                "duplicate_indicator": 0,
            },
            {
                "claim_id": "CLM-B2",
                "procedure_code": "27447",
                "claim_amount": 9000.0,
                "duplicate_indicator": 1,
            },
        ],
        "prioritize_by": "risk_score",
    }

    response = client.post("/api/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["claims"]) == 2


def test_copilot_summary_endpoint() -> None:
    """Verify POST /api/copilot/summary generates grounded audit memo."""
    payload = {
        "claim": {
            "claim_id": "CLM-COP-01",
            "provider_id": "PRV-505",
            "procedure_code": "99285",
            "claim_amount": 1400.0,
            "submission_delay": 85,
            "missing_fields": 1,
            "duplicate_indicator": 1,
            "provider_claim_frequency": 300,
            "historical_rejection_rate": 0.25,
            "amount_deviation": 2.2,
        }
    }

    response = client.post("/api/copilot/summary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary_markdown" in data
    assert "ClaimRisk AI Analyst Review Memo" in data["summary_markdown"]


def test_copilot_chat_endpoint() -> None:
    """Verify POST /api/copilot/chat answers auditor inquiries."""
    payload = {
        "question": "Why was this claim flagged?",
        "claim": {
            "claim_id": "CLM-CHAT-01",
            "procedure_code": "99214",
            "claim_amount": 500.0,
            "submission_delay": 40,
            "duplicate_indicator": 0,
            "historical_rejection_rate": 0.10,
        },
    }

    response = client.post("/api/copilot/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10


def test_model_comparison_endpoint() -> None:
    """Verify GET /api/models/comparison returns benchmark metrics."""
    response = client.get("/api/models/comparison")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model" in data


def test_explain_global_endpoint() -> None:
    """Verify GET /api/explain/global returns feature importance rankings."""
    response = client.get("/api/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance_ranking" in data


def test_dashboard_endpoint() -> None:
    """Verify GET / serves HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ClaimRisk AI" in response.text
