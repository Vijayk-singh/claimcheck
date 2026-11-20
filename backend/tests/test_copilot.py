"""
Tests for Grounded Analyst Copilot Module.
"""

from __future__ import annotations

import pytest

from backend.ml.copilot import ClaimRiskCopilot


def test_copilot_generate_grounded_summary() -> None:
    """Verify grounded memo generation contains all expected clinical & audit sections."""
    copilot = ClaimRiskCopilot()

    claim_data = {
        "claim_id": "CLM-COPILOT-1",
        "provider_id": "PRV-2002",
        "procedure_code": "99285",
        "claim_amount": 1200.0,
        "submission_delay": 80,
        "missing_fields": 2,
        "duplicate_indicator": 1,
        "historical_rejection_rate": 0.22,
        "amount_deviation": 2.1,
    }

    prediction = {
        "risk_score": 0.88,
        "risk_tier": "HIGH",
        "expected_financial_exposure": 1056.0,
        "recommended_action": "Priority manual review",
        "explanation": {
            "top_risk_drivers": [
                {"display_name": "Procedure Fee Deviation (Z-Score)", "shap_value": 0.15, "value": 2.1},
                {"display_name": "Suspected Duplicate Submission", "shap_value": 0.12, "value": 1.0},
            ],
            "top_mitigating_factors": [],
        },
    }

    res = copilot.generate_grounded_summary(claim_data, prediction)

    assert res["claim_id"] == "CLM-COPILOT-1"
    assert res["risk_tier"] == "HIGH"
    assert "summary_markdown" in res

    md = res["summary_markdown"]
    assert "ClaimRisk AI Analyst Review Memo" in md
    assert "Executive Risk Assessment" in md
    assert "Model Feature Attributions" in md
    assert "Auditor Action Checklist" in md
    assert "Duplicate Cross-Check" in md


def test_copilot_answer_questions() -> None:
    """Verify grounded answers to common auditor inquiries."""
    copilot = ClaimRiskCopilot()

    claim_data = {
        "claim_id": "CLM-COPILOT-2",
        "provider_id": "PRV-3003",
        "procedure_code": "99214",
        "claim_amount": 600.0,
        "submission_delay": 30,
        "duplicate_indicator": 1,
        "historical_rejection_rate": 0.25,
    }
    prediction = {
        "risk_score": 0.91,
        "risk_tier": "HIGH",
        "expected_financial_exposure": 546.0,
        "recommended_action": "Immediate hold: Review suspected duplicate",
    }

    # Test why flagged
    ans1 = copilot.answer_question("Why was this claim flagged?", claim_data, prediction)
    assert "HIGH" in ans1

    # Test fee inquiry
    ans2 = copilot.answer_question("Is the fee markup excessive?", claim_data, prediction)
    assert "benchmark" in ans2.lower()

    # Test duplicate inquiry
    ans3 = copilot.answer_question("Is there a duplicate issue?", claim_data, prediction)
    assert "Duplicate" in ans3
