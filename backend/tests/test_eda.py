"""
Unit and integration tests for ClaimRisk AI Exploratory Data Analysis module (Phase 2).

Verifies:
- Dataset profiling (dimensions, null counts, stats, categorical distributions).
- Data quality auditing (financial constraints, dates, delay mismatches, outliers).
- Detection of corrupted data distributions.
- Generation of all 8 Matplotlib visualization artifacts.
- Generation and structure of JSON summary and Markdown reports.
- CLI argument parsing and end-to-end execution.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backend.ml.eda import (
    analyze_data_quality,
    generate_eda_report,
    generate_visualizations,
    main as eda_main,
    parse_arguments as eda_parse_args,
    profile_dataset,
    run_eda,
)
from backend.ml.generate_data import SyntheticClaimGenerator


@pytest.fixture
def sample_claims_df() -> pd.DataFrame:
    """Generate 600 synthetic claims for fast testing."""
    gen = SyntheticClaimGenerator(seed=42)
    return gen.generate_dataset(num_records=600)


def test_profile_dataset(sample_claims_df: pd.DataFrame):
    """Verify profile_dataset produces valid profiling dictionary."""
    profile = profile_dataset(sample_claims_df)

    assert profile["dimensions"]["rows"] == 600
    assert profile["dimensions"]["columns"] == 14
    assert profile["missing_analysis"]["total_missing_cells"] == 0
    assert profile["duplicate_analysis"]["claim_id_unique"] is True
    assert profile["duplicate_analysis"]["duplicate_full_rows"] == 0

    assert "claim_amount" in profile["numeric_statistics"]
    assert "procedure_code" in profile["categorical_statistics"]
    assert "target_distribution" in profile
    assert profile["target_distribution"]["positive_class_percentage"] > 0


def test_analyze_data_quality_clean(sample_claims_df: pd.DataFrame):
    """Verify data quality audit marks synthetic claims dataset as clean."""
    quality = analyze_data_quality(sample_claims_df)

    assert quality["overall_quality_status"] == "CLEAN"
    assert quality["financial_integrity"]["is_valid"] is True
    assert quality["temporal_integrity"]["is_valid"] is True
    assert quality["financial_integrity"]["negative_or_zero_claim_amounts"] == 0
    assert quality["temporal_integrity"]["submission_precedes_service_count"] == 0


def test_analyze_data_quality_detects_corruptions(sample_claims_df: pd.DataFrame):
    """Verify data quality audit catches financial, temporal, and delay corruptions."""
    corrupted_df = sample_claims_df.copy()

    # Corrupt claim amount to negative
    corrupted_df.loc[0, "claim_amount"] = -150.0

    # Corrupt submission date to precede service date
    corrupted_df.loc[1, "submission_date"] = "2020-01-01"
    corrupted_df.loc[1, "service_date"] = "2023-01-01"

    # Corrupt submission delay
    corrupted_df.loc[2, "submission_delay"] = 999

    quality = analyze_data_quality(corrupted_df)

    assert quality["overall_quality_status"] == "ANOMALOUS"
    assert quality["financial_integrity"]["is_valid"] is False
    assert quality["financial_integrity"]["negative_or_zero_claim_amounts"] > 0
    assert quality["temporal_integrity"]["submission_precedes_service_count"] > 0
    assert quality["temporal_integrity"]["submission_delay_calculation_mismatches"] > 0


def test_generate_visualizations(sample_claims_df: pd.DataFrame, tmp_path: Path):
    """Verify all 8 visualization PNG files are created and non-empty."""
    out_dir = tmp_path / "eda_charts"
    generated_plots = generate_visualizations(sample_claims_df, output_dir=out_dir)

    expected_plot_names = [
        "target_distribution.png",
        "claim_amount_distribution.png",
        "submission_delay_distribution.png",
        "amount_deviation_distribution.png",
        "missing_fields_distribution.png",
        "numeric_correlation_heatmap.png",
        "features_by_target.png",
        "risk_indicators_by_target.png",
    ]

    assert len(generated_plots) == 8
    for name in expected_plot_names:
        plot_path = out_dir / name
        assert plot_path.exists(), f"Plot not found: {name}"
        assert plot_path.stat().st_size > 1000, f"Plot file unexpectedly small: {name}"


def test_generate_eda_report(sample_claims_df: pd.DataFrame, tmp_path: Path):
    """Verify JSON summary and Markdown reports are generated properly."""
    profile = profile_dataset(sample_claims_df)
    quality = analyze_data_quality(sample_claims_df)

    out_dir = tmp_path / "reports"
    json_path, md_path = generate_eda_report(profile, quality, output_dir=out_dir)

    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON structure
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "profiling" in data
    assert "data_quality" in data

    # Verify Markdown contains executive observations
    md_text = md_path.read_text(encoding="utf-8")
    assert "ClaimRisk AI" in md_text
    assert "Executive Summary & Key Observations" in md_text
    assert "Strategic Recommendations" in md_text


def test_run_eda_end_to_end(sample_claims_df: pd.DataFrame, tmp_path: Path):
    """Verify run_eda executes end-to-end and returns expected structure."""
    csv_file = tmp_path / "claims_sample.csv"
    sample_claims_df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "full_eda_run"
    results = run_eda(data_path=csv_file, output_dir=out_dir)

    assert "profiling" in results
    assert "data_quality" in results
    assert "visualizations" in results
    assert len(results["visualizations"]) == 8
    assert (out_dir / "eda_summary.json").exists()
    assert (out_dir / "eda_report.md").exists()


def test_cli_argument_parsing():
    """Verify EDA CLI argument parser accepts custom input and output-dir."""
    test_args = ["--input", "custom_claims.csv", "--output-dir", "custom_reports"]
    with patch("sys.argv", ["eda.py"] + test_args):
        parsed = eda_parse_args()
        assert str(parsed.input) == "custom_claims.csv"
        assert str(parsed.output_dir) == "custom_reports"
