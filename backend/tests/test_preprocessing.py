"""
Unit and integration tests for ClaimRisk AI Preprocessing module (Phase 2).

Verifies:
- Correct feature matrix X and target y separation.
- Strict prevention of target leakage (requires_review excluded from X).
- Strict exclusion of post-adjudication outcome (approved_amount excluded from X).
- Strict exclusion of nominal identifiers (claim_id, provider_id).
- Strict exclusion of raw date timestamps (service_date, submission_date).
- Categorical feature encoding (OneHotEncoder for procedure_code).
- Robust handling of unseen / unknown categories during inference.
- Robust handling of unexpected missing/null values via imputation.
- Stratified train/test partitioning and class distribution preservation.
- Preprocessing fit solely on training data (no lookahead / data leakage).
- Deterministic reproducibility under fixed random seed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from unittest.mock import patch

from backend.ml.generate_data import SyntheticClaimGenerator
from backend.ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    EXCLUDED_COLUMNS,
    FULL_NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    fit_transform_data,
    get_feature_columns,
    main as prep_main,
    parse_arguments as prep_parse_args,
    prepare_features,
    preprocess_claims,
    split_dataset,
)


@pytest.fixture
def sample_claims() -> pd.DataFrame:
    """Generate 1,000 synthetic claims for fast, deterministic unit testing."""
    gen = SyntheticClaimGenerator(seed=42)
    return gen.generate_dataset(num_records=1000)


def test_get_feature_columns():
    """Verify get_feature_columns properly categorizes all feature groups."""
    cols = get_feature_columns()
    assert "numerical" in cols
    assert "categorical" in cols
    assert "excluded" in cols
    assert "target" in cols

    assert TARGET_COLUMN in cols["target"]
    assert "procedure_code" in cols["categorical"]
    assert "approved_amount" in cols["excluded"]
    assert "claim_id" in cols["excluded"]
    assert "provider_id" in cols["excluded"]


def test_target_separation_and_exclusion(sample_claims: pd.DataFrame):
    """Verify target separation and strict absence of leakage columns in X."""
    X, y = prepare_features(sample_claims, perform_engineering=True)

    # 1. Target y is separated and binary
    assert y is not None
    assert len(y) == len(sample_claims)
    assert set(y.unique()).issubset({0, 1})

    # 2. Target strictly excluded from X
    assert TARGET_COLUMN not in X.columns

    # 3. Post-adjudication outcome approved_amount strictly excluded
    assert "approved_amount" not in X.columns

    # 4. Identifiers strictly excluded
    assert "claim_id" not in X.columns
    assert "provider_id" not in X.columns

    # 5. Raw date strings excluded
    assert "service_date" not in X.columns
    assert "submission_date" not in X.columns


def test_stratified_split_preserves_class_ratio(sample_claims: pd.DataFrame):
    """Verify stratified split preserves target positive class percentage."""
    X, y = prepare_features(sample_claims, perform_engineering=True)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)

    assert len(X_train) == 800
    assert len(X_test) == 200

    overall_pos_rate = float(y.mean())
    train_pos_rate = float(y_train.mean())
    test_pos_rate = float(y_test.mean())

    # Proportions should match within 1 percentage point under stratification
    assert abs(train_pos_rate - overall_pos_rate) < 0.015
    assert abs(test_pos_rate - overall_pos_rate) < 0.015


def test_fit_solely_on_training_data(sample_claims: pd.DataFrame):
    """Verify preprocessor parameters (scaler means) are learned strictly from X_train."""
    X, y = prepare_features(sample_claims, perform_engineering=True)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)

    preprocessor = build_preprocessor()
    X_train_proc, X_test_proc, fitted_preproc, feature_names = fit_transform_data(
        preprocessor, X_train, X_test
    )

    # Inspect numerical scaler means fitted inside the ColumnTransformer
    num_pipeline = fitted_preproc.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scaler"]
    num_cols = FULL_NUMERICAL_COLUMNS

    # Mean of claim_amount learned by scaler should match X_train mean, NOT full X mean
    claim_amt_idx = num_cols.index("claim_amount")
    learned_mean = scaler.mean_[claim_amt_idx]
    train_mean = X_train["claim_amount"].mean()
    full_mean = X["claim_amount"].mean()

    assert learned_mean == pytest.approx(train_mean, rel=1e-5)
    # Train mean is statistically distinct from full mean
    assert learned_mean != pytest.approx(full_mean, rel=1e-7)


def test_categorical_one_hot_encoding(sample_claims: pd.DataFrame):
    """Verify procedure_code is one-hot encoded into discrete binary columns."""
    X, y = prepare_features(sample_claims, perform_engineering=True)
    X_train, X_test, _, _ = split_dataset(X, y, test_size=0.20, random_state=42)

    preprocessor = build_preprocessor()
    X_train_proc, X_test_proc, fitted_preproc, feature_names = fit_transform_data(
        preprocessor, X_train, X_test
    )

    # Check that procedure_code categories appear in feature_names
    procedure_features = [f for f in feature_names if "procedure_code" in f]
    assert len(procedure_features) > 1

    # Check that transformed outputs are 2D arrays with valid dimensions
    assert X_train_proc.ndim == 2
    assert X_train_proc.shape[1] == len(feature_names)
    assert X_test_proc.shape[1] == len(feature_names)


def test_handling_unknown_categories(sample_claims: pd.DataFrame):
    """Verify OneHotEncoder with handle_unknown='ignore' handles new categories cleanly."""
    X, y = prepare_features(sample_claims, perform_engineering=True)
    X_train, X_test, _, _ = split_dataset(X, y, test_size=0.20, random_state=42)

    # Introduce unknown procedure codes into test set
    X_test_unseen = X_test.copy()
    X_test_unseen.iloc[0, X_test_unseen.columns.get_loc("procedure_code")] = "99999_NOVEL_CODE"

    preprocessor = build_preprocessor()
    fitted_preproc = preprocessor.fit(X_train)

    # Transforming unseen categories must not crash
    transformed_test = fitted_preproc.transform(X_test_unseen)
    assert not np.isnan(transformed_test).any()
    assert transformed_test.shape[0] == len(X_test_unseen)


def test_handling_missing_values(sample_claims: pd.DataFrame):
    """Verify SimpleImputer handles missing/NaN values without failing."""
    X, y = prepare_features(sample_claims, perform_engineering=True)
    X_train, X_test, _, _ = split_dataset(X, y, test_size=0.20, random_state=42)

    # Inject NaNs into test numerical and categorical columns
    X_test_with_nans = X_test.copy()
    X_test_with_nans.loc[X_test_with_nans.index[0], "claim_amount"] = np.nan
    X_test_with_nans.loc[X_test_with_nans.index[1], "procedure_code"] = np.nan

    preprocessor = build_preprocessor()
    fitted_preproc = preprocessor.fit(X_train)

    # Transform test set with missing values
    transformed = fitted_preproc.transform(X_test_with_nans)
    assert not np.isnan(transformed).any(), "Found unexpected NaNs in transformed output"


def test_reproducibility(sample_claims: pd.DataFrame):
    """Verify identical random seeds produce identical partitioned splits."""
    X, y = prepare_features(sample_claims, perform_engineering=True)

    X_tr1, X_te1, y_tr1, y_te1 = split_dataset(X, y, test_size=0.20, random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = split_dataset(X, y, test_size=0.20, random_state=42)

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(X_te1, X_te2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    pd.testing.assert_series_equal(y_te1, y_te2)


def test_preprocess_claims_end_to_end(tmp_path):
    """Verify end-to-end preprocess_claims pipeline execution and artifact persistence."""
    # Write a small CSV to tmp_path
    gen = SyntheticClaimGenerator(seed=101)
    df = gen.generate_dataset(num_records=500)
    csv_file = tmp_path / "claims.csv"
    df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "processed_output"
    results = preprocess_claims(data_path=csv_file, output_dir=out_dir, test_size=0.20, random_state=42)

    assert "X_train_processed" in results
    assert "X_test_processed" in results
    assert "preprocessor" in results
    assert "feature_names" in results

    # Verify persisted artifacts
    assert (out_dir / "X_train_processed.csv").exists()
    assert (out_dir / "X_test_processed.csv").exists()
    assert (out_dir / "y_train.csv").exists()
    assert (out_dir / "y_test.csv").exists()
    assert (out_dir / "feature_names.json").exists()
    assert (out_dir / "preprocessor.joblib").exists()


def test_prep_cli_parsing():
    """Verify preprocessing CLI argument parsing."""
    test_args = [
        "--input", "my_claims.csv",
        "--output-dir", "my_processed",
        "--test-size", "0.25",
        "--seed", "99",
    ]
    with patch("sys.argv", ["prep.py"] + test_args):
        args = prep_parse_args()
        assert str(args.input) == "my_claims.csv"
        assert str(args.output_dir) == "my_processed"
        assert args.test_size == 0.25
        assert args.seed == 99


def test_prep_cli_execution(tmp_path):
    """Verify preprocessing CLI main execution."""
    gen = SyntheticClaimGenerator(seed=202)
    df = gen.generate_dataset(num_records=400)
    csv_file = tmp_path / "claims_input.csv"
    df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "cli_processed_dir"
    test_args = [
        "--input", str(csv_file),
        "--output-dir", str(out_dir),
        "--test-size", "0.20",
        "--seed", "42",
    ]
    with patch("sys.argv", ["prep.py"] + test_args):
        prep_main()

    assert (out_dir / "X_train_processed.csv").exists()
    assert (out_dir / "preprocessor.joblib").exists()

