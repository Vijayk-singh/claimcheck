"""
Unit and Integration Tests for ClaimRisk AI Synthetic Claims Generator.

Verifies:
- Schema conformance
- Probabilistic target integrity
- Non-determinism & lack of single-feature correlation leaks
- Reproducibility under identical random seeds
- Value ranges and date consistency
- Custom validation utility
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.ml.generate_data import (
    PROCEDURE_CATALOG,
    SyntheticClaimGenerator,
    main as generate_main,
    parse_arguments as generate_parse_args,
    validate_claims_dataset,
)
from backend.ml.validate_data import main as validate_main, parse_arguments as validate_parse_args


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Generate a quick 1,000-row sample for fast unit tests."""
    generator = SyntheticClaimGenerator(seed=42)
    return generator.generate_dataset(num_records=1000)


def test_schema_columns(sample_dataset: pd.DataFrame):
    """Verify all 14 required columns are present in the generated DataFrame."""
    expected_columns = [
        "claim_id",
        "provider_id",
        "procedure_code",
        "claim_amount",
        "approved_amount",
        "service_date",
        "submission_date",
        "submission_delay",
        "missing_fields",
        "duplicate_indicator",
        "provider_claim_frequency",
        "historical_rejection_rate",
        "amount_deviation",
        "requires_review",
    ]
    assert list(sample_dataset.columns) == expected_columns


def test_dataset_size():
    """Verify generator respects requested record counts."""
    gen = SyntheticClaimGenerator(seed=123)
    df_small = gen.generate_dataset(num_records=250)
    assert len(df_small) == 250


def test_invalid_size_raises_value_error():
    """Verify generator rejects negative or zero sizes."""
    gen = SyntheticClaimGenerator(seed=42)
    with pytest.raises(ValueError):
        gen.generate_dataset(num_records=0)
    with pytest.raises(ValueError):
        gen.generate_dataset(num_records=-50)


def test_reproducibility():
    """Verify identical seeds produce identical datasets."""
    gen1 = SyntheticClaimGenerator(seed=999)
    df1 = gen1.generate_dataset(num_records=300)

    gen2 = SyntheticClaimGenerator(seed=999)
    df2 = gen2.generate_dataset(num_records=300)

    pd.testing.assert_frame_equal(df1, df2)


def test_no_null_values(sample_dataset: pd.DataFrame):
    """Verify zero missing/null values across the dataset."""
    assert sample_dataset.isnull().sum().sum() == 0


def test_unique_claim_ids(sample_dataset: pd.DataFrame):
    """Verify claim_id values are strictly unique primary keys."""
    assert sample_dataset["claim_id"].nunique() == len(sample_dataset)


def test_procedure_codes_valid(sample_dataset: pd.DataFrame):
    """Verify procedure codes belong exclusively to the defined catalog."""
    catalog_codes = {p.code for p in PROCEDURE_CATALOG}
    dataset_codes = set(sample_dataset["procedure_code"].unique())
    assert dataset_codes.issubset(catalog_codes)


def test_financial_sanity(sample_dataset: pd.DataFrame):
    """Verify financial constraints: positive claims and non-negative approvals <= billed."""
    assert (sample_dataset["claim_amount"] > 0).all()
    assert (sample_dataset["approved_amount"] >= 0).all()
    assert (sample_dataset["approved_amount"] <= sample_dataset["claim_amount"]).all()


def test_date_and_delay_consistency(sample_dataset: pd.DataFrame):
    """Verify submission_delay equals the exact difference between submission_date and service_date."""
    s_date = pd.to_datetime(sample_dataset["service_date"])
    sub_date = pd.to_datetime(sample_dataset["submission_date"])
    computed_delay = (sub_date - s_date).dt.days

    assert (sample_dataset["submission_delay"] >= 0).all()
    assert (computed_delay == sample_dataset["submission_delay"]).all()


def test_duplicate_and_target_flags(sample_dataset: pd.DataFrame):
    """Verify indicator fields are binary {0, 1}."""
    assert set(sample_dataset["duplicate_indicator"].unique()).issubset({0, 1})
    assert set(sample_dataset["requires_review"].unique()).issubset({0, 1})


def test_provider_rejection_rate_bounds(sample_dataset: pd.DataFrame):
    """Verify historical rejection rate is a valid probability bounded in [0.0, 1.0]."""
    assert (sample_dataset["historical_rejection_rate"] >= 0.0).all()
    assert (sample_dataset["historical_rejection_rate"] <= 1.0).all()


def test_non_deterministic_target_assignment(sample_dataset: pd.DataFrame):
    """
    Verify target is probabilistic:
    1. Both classes 0 and 1 exist.
    2. No single feature has a trivial 1.0 correlation with target (no single-feature determinism).
    """
    review_counts = sample_dataset["requires_review"].value_counts()
    assert 0 in review_counts.index
    assert 1 in review_counts.index

    # Check correlations
    numeric_cols = sample_dataset.select_dtypes(include=["number"]).columns
    correlations = sample_dataset[numeric_cols].corr()["requires_review"]

    for col, corr in correlations.items():
        if col != "requires_review":
            assert abs(corr) < 0.90, f"Feature {col} has suspiciously high correlation {corr} with target"


def test_validation_function_detects_corrupted_data(sample_dataset: pd.DataFrame):
    """Verify validate_claims_dataset returns True on clean data and False on mutated data."""
    is_valid, issues = validate_claims_dataset(sample_dataset)
    assert is_valid is True
    assert len(issues) == 0

    # Corrupt a copy by introducing negative claim amount
    corrupted = sample_dataset.copy()
    corrupted.loc[0, "claim_amount"] = -100.0
    is_valid_bad, bad_issues = validate_claims_dataset(corrupted)
    assert is_valid_bad is False
    assert any("claim_amount contains non-positive values" in msg for msg in bad_issues)


def test_cli_argument_parsing():
    """Verify CLI argument defaults and custom inputs."""
    with patch("sys.argv", ["generate_data.py", "--size", "50", "--seed", "7", "--output", "test.csv"]):
        args = generate_parse_args()
        assert args.size == 50
        assert args.seed == 7
        assert str(args.output) == "test.csv"

    with patch("sys.argv", ["validate_data.py", "--input", "test.csv"]):
        v_args = validate_parse_args()
        assert str(v_args.input) == "test.csv"


def test_cli_execution_end_to_end(tmp_path: Path):
    """Verify CLI generation and validation end-to-end with temporary files."""
    test_out = tmp_path / "cli_test_claims.csv"

    # Test generator CLI
    with patch("sys.argv", ["generate_data.py", "--size", "100", "--seed", "42", "--output", str(test_out)]):
        generate_main()
    assert test_out.exists()

    # Test validator CLI
    with patch("sys.argv", ["validate_data.py", "--input", str(test_out)]):
        with pytest.raises(SystemExit) as exit_info:
            validate_main()
        assert exit_info.value.code == 0

