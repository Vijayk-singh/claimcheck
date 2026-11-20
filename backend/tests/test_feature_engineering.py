"""
Unit tests for ClaimRisk AI Feature Engineering module (Phase 2).

Verifies:
- All 9 engineered features generated correctly.
- Preservation of original dataset (defensive copying).
- Benchmark ratio calculations and graceful unknown code fallbacks.
- Temporal date extraction (month, day of week, weekend).
- Threshold risk indicators logic and edge bounds.
- Absence of target leakage (requires_review not needed or used).
- Robust handling of edge cases and validation error detection.
"""

import numpy as np
import pandas as pd
import pytest

from unittest.mock import patch

from backend.ml.feature_engineering import (
    ENGINEERED_FEATURE_NAMES,
    PROCEDURE_BENCHMARKS,
    compute_benchmark_ratio,
    engineer_features,
    extract_date_features,
    extract_threshold_indicators,
    main as fe_main,
    parse_arguments as fe_parse_args,
    validate_engineered_features,
)


@pytest.fixture
def mini_claims_df() -> pd.DataFrame:
    """Provide a representative sample claims DataFrame."""
    return pd.DataFrame(
        {
            "claim_id": ["CLM-001", "CLM-002", "CLM-003", "CLM-004"],
            "provider_id": ["PRV-001", "PRV-002", "PRV-003", "PRV-004"],
            "procedure_code": ["99213", "99215", "70450", "27447"],
            "claim_amount": [110.0, 450.0, 430.0, 8400.0],
            "approved_amount": [95.0, 200.0, 390.0, 4000.0],
            "service_date": ["2023-01-01", "2023-06-15", "2023-09-02", "2023-12-25"],  # Sun, Thu, Sat, Mon
            "submission_date": ["2023-01-15", "2023-08-25", "2023-09-12", "2024-01-05"],
            "submission_delay": [14, 71, 10, 11],
            "missing_fields": [0, 2, 1, 3],
            "duplicate_indicator": [0, 1, 0, 0],
            "provider_claim_frequency": [85, 320, 120, 450],
            "historical_rejection_rate": [0.05, 0.28, 0.12, 0.22],
            "amount_deviation": [0.0, 6.43, 0.0, 8.08],
            "requires_review": [0, 1, 0, 1],
        }
    )


def test_feature_generation_columns(mini_claims_df: pd.DataFrame):
    """Verify all 9 engineered features are added to DataFrame."""
    result = engineer_features(mini_claims_df)
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in result.columns, f"Missing expected engineered feature: {col}"
    assert len(result) == len(mini_claims_df)


def test_original_dataframe_unmutated(mini_claims_df: pd.DataFrame):
    """Verify defensive copying: original DataFrame is not modified in-place."""
    original_cols = list(mini_claims_df.columns)
    _ = engineer_features(mini_claims_df)
    assert list(mini_claims_df.columns) == original_cols


def test_benchmark_ratio_calculation(mini_claims_df: pd.DataFrame):
    """Verify procedure benchmark ratio calculation matches claim_amount / benchmark_mean."""
    ratios = compute_benchmark_ratio(mini_claims_df)
    # 99213 benchmark is 110.0 -> 110.0 / 110.0 = 1.0
    assert ratios.iloc[0] == pytest.approx(1.0, rel=1e-3)
    # 99215 benchmark is 225.0 -> 450.0 / 225.0 = 2.0
    assert ratios.iloc[1] == pytest.approx(2.0, rel=1e-3)
    # 70450 benchmark is 430.0 -> 430.0 / 430.0 = 1.0
    assert ratios.iloc[2] == pytest.approx(1.0, rel=1e-3)
    # 27447 benchmark is 4200.0 -> 8400.0 / 4200.0 = 2.0
    assert ratios.iloc[3] == pytest.approx(2.0, rel=1e-3)


def test_benchmark_ratio_unknown_code_fallback():
    """Verify compute_benchmark_ratio gracefully falls back for unseen procedure codes."""
    df_unknown = pd.DataFrame(
        {
            "procedure_code": ["UNKNOWN_999", "UNKNOWN_999"],
            "claim_amount": [200.0, 300.0],
        }
    )
    ratios = compute_benchmark_ratio(df_unknown)
    assert not ratios.isnull().any()
    assert (ratios > 0).all()


def test_date_feature_extraction():
    """Verify date feature parsing correctly extracts month, day of week, and weekend status."""
    df_dates = pd.DataFrame(
        {
            "service_date": [
                "2023-01-01",  # Sunday (month=1, dow=6, weekend=1)
                "2023-01-02",  # Monday (month=1, dow=0, weekend=0)
                "2023-07-15",  # Saturday (month=7, dow=5, weekend=1)
                "2023-11-22",  # Wednesday (month=11, dow=2, weekend=0)
            ]
        }
    )
    date_features = extract_date_features(df_dates, date_col="service_date")
    assert date_features["service_month"].tolist() == [1, 1, 7, 11]
    assert date_features["service_day_of_week"].tolist() == [6, 0, 5, 2]
    assert date_features["service_is_weekend"].tolist() == [1, 0, 1, 0]


def test_threshold_indicators(mini_claims_df: pd.DataFrame):
    """Verify threshold indicators evaluate strictly at the boundary values."""
    indicators = extract_threshold_indicators(
        mini_claims_df,
        delay_threshold=60,
        missing_fields_threshold=2,
        rejection_rate_threshold=0.20,
        claim_frequency_threshold=300,
        amount_deviation_threshold=2.0,
    )
    # Row 0: delay=14, missing=0, rej=0.05, freq=85, dev=0.0 -> all 0
    assert indicators.iloc[0].to_dict() == {
        "high_submission_delay": 0,
        "has_multiple_missing_fields": 0,
        "high_provider_rejection_rate": 0,
        "high_provider_claim_frequency": 0,
        "high_amount_deviation": 0,
    }
    # Row 1: delay=71 (>60), missing=2 (>=2), rej=0.28 (>0.20), freq=320 (>300), dev=6.43 (>2.0) -> all 1
    assert indicators.iloc[1].to_dict() == {
        "high_submission_delay": 1,
        "has_multiple_missing_fields": 1,
        "high_provider_rejection_rate": 1,
        "high_provider_claim_frequency": 1,
        "high_amount_deviation": 1,
    }


def test_no_target_leakage_in_features(mini_claims_df: pd.DataFrame):
    """Verify features can be engineered on data without target variable 'requires_review'."""
    df_no_target = mini_claims_df.drop(columns=["requires_review"])
    result = engineer_features(df_no_target)
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in result.columns
    assert "requires_review" not in result.columns


def test_empty_dataframe_handling():
    """Verify feature engineering handles empty DataFrame cleanly without crashing."""
    empty_df = pd.DataFrame(columns=["claim_amount", "procedure_code", "service_date"])
    result = engineer_features(empty_df)
    assert len(result) == 0
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in result.columns


def test_validation_catches_invalid_features(mini_claims_df: pd.DataFrame):
    """Verify validation detects corrupt engineered values."""
    df_engineered = engineer_features(mini_claims_df)

    # Corrupt service_month to 13
    corrupt_month = df_engineered.copy()
    corrupt_month["service_month"] = 13
    with pytest.raises(ValueError, match="service_month"):
        validate_engineered_features(corrupt_month)

    # Corrupt binary indicator to 5
    corrupt_binary = df_engineered.copy()
    corrupt_binary["service_is_weekend"] = 5
    with pytest.raises(ValueError, match="Binary feature"):
        validate_engineered_features(corrupt_binary)


def test_fe_cli_parsing():
    """Verify feature engineering CLI argument parsing."""
    test_args = ["--input", "custom_in.csv", "--output", "custom_out.csv"]
    with patch("sys.argv", ["fe.py"] + test_args):
        args = fe_parse_args()
        assert str(args.input) == "custom_in.csv"
        assert str(args.output) == "custom_out.csv"


def test_fe_cli_execution_end_to_end(mini_claims_df: pd.DataFrame, tmp_path):
    """Verify feature engineering main CLI execution."""
    in_file = tmp_path / "claims_input.csv"
    out_file = tmp_path / "claims_out.csv"
    mini_claims_df.to_csv(in_file, index=False)

    test_args = ["--input", str(in_file), "--output", str(out_file)]
    with patch("sys.argv", ["fe.py"] + test_args):
        fe_main()

    assert out_file.exists()
    saved_df = pd.read_csv(out_file)
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in saved_df.columns

