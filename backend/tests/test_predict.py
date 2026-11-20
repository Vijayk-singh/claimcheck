"""
Tests for Prediction & Prioritization Pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.predict import ClaimRiskScorer, predict_claim_risk


def test_scorer_predict_single() -> None:
    """Verify single claim scoring and output schema."""
    sample_claim = {
        "claim_id": "CLM-UNIT-01",
        "provider_id": "PRV-1001",
        "procedure_code": "99214",
        "claim_amount": 350.0,
        "submission_delay": 45,
        "missing_fields": 1,
        "duplicate_indicator": 0,
        "historical_rejection_rate": 0.12,
        "amount_deviation": 0.5,
    }

    scorer = ClaimRiskScorer()
    res = scorer.predict_single(sample_claim, include_explanation=True)

    assert res["claim_id"] == "CLM-UNIT-01"
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}
    assert res["expected_financial_exposure"] >= 0.0
    assert "recommended_action" in res
    assert "explanation" in res


def test_scorer_predict_batch_prioritization() -> None:
    """Verify batch scoring and sorting by priority queue."""
    claims = [
        {"claim_id": "CLM-A", "procedure_code": "99214", "claim_amount": 100.0, "duplicate_indicator": 0},
        {"claim_id": "CLM-B", "procedure_code": "27447", "claim_amount": 8000.0, "duplicate_indicator": 1},
    ]
    df = pd.DataFrame(claims)

    scorer = ClaimRiskScorer()
    scored_df = scorer.predict_batch(df, prioritize_by="risk_score")

    assert len(scored_df) == 2
    assert "risk_score" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert "financial_exposure" in scored_df.columns
    assert scored_df.iloc[0]["risk_score"] >= scored_df.iloc[1]["risk_score"]
