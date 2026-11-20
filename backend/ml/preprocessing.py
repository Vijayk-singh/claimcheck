"""
ClaimRisk AI - Data Preprocessing Module.

Prepares raw claims data for supervised machine learning by implementing:
- Clean isolation of predictive features X from target y ('requires_review').
- Strict exclusion of post-adjudication outcome ('approved_amount') to prevent data leakage.
- Exclusion of nominal identifiers ('claim_id', 'provider_id').
- Exclusion of raw date strings after feature engineering ('service_date', 'submission_date').
- Categorical one-hot encoding for 'procedure_code' with unknown category handling.
- Numerical missing value imputation and standard scaling.
- Scikit-learn Pipeline and ColumnTransformer integration.
- Stratified 80:20 train/test split.
- Strict compliance: fits all transformers solely on training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.feature_engineering import (
    ENGINEERED_FEATURE_NAMES,
    ClaimFeatureTransformer,
    ProcedureBenchmarkTransformer,
    engineer_features,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.preprocessing")

# Column Definitions
TARGET_COLUMN: str = "requires_review"

EXCLUDED_COLUMNS: List[str] = [
    "claim_id",           # High-cardinality primary key identifier
    "provider_id",        # Identifier; provider risk already encapsulated in numerical metrics
    "approved_amount",    # Post-adjudication outcome (STRICT TARGET LEAKAGE PREVENTION)
    "service_date",       # Raw timestamp replaced by engineered temporal signals
    "submission_date",    # Raw timestamp replaced by submission_delay and temporal signals
]

CATEGORICAL_COLUMNS: List[str] = [
    "procedure_code",     # Clinical procedure category (CPT/HCPCS proxies)
]

BASE_NUMERICAL_COLUMNS: List[str] = [
    "claim_amount",
    "submission_delay",
    "missing_fields",
    "duplicate_indicator",
    "provider_claim_frequency",
    "historical_rejection_rate",
    "amount_deviation",
]

# Numerical features include original metrics plus newly engineered continuous & indicator features
FULL_NUMERICAL_COLUMNS: List[str] = BASE_NUMERICAL_COLUMNS + ENGINEERED_FEATURE_NAMES


def get_feature_columns() -> Dict[str, List[str]]:
    """
    Return dictionary of categorized column groups for data preprocessing.

    Returns:
        Dict[str, List[str]]: Dictionary with 'numerical', 'categorical', 'excluded', and 'target'.
    """
    return {
        "numerical": list(FULL_NUMERICAL_COLUMNS),
        "categorical": list(CATEGORICAL_COLUMNS),
        "excluded": list(EXCLUDED_COLUMNS),
        "target": [TARGET_COLUMN],
    }


def prepare_features(
    df: pd.DataFrame,
    perform_engineering: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare feature matrix X and target vector y from input DataFrame.

    - Performs feature engineering if required engineered columns are missing.
    - Isolates target variable 'requires_review' if present.
    - Strictly drops excluded columns ('approved_amount', 'claim_id', etc.).
    - Casts categorical features appropriately.

    Args:
        df: Input DataFrame containing claim records.
        perform_engineering: Whether to automatically run feature engineering.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.Series]]: (X features, y target or None)
    """
    if df.empty:
        raise ValueError("Cannot prepare features on an empty DataFrame.")

    # Defensive copy
    working_df = df.copy()

    # 1. Feature Engineering step
    if perform_engineering:
        missing_engineered = [c for c in ENGINEERED_FEATURE_NAMES if c not in working_df.columns]
        if missing_engineered:
            logger.info("Executing feature engineering inside prepare_features...")
            working_df = engineer_features(working_df)

    # 2. Extract target if present
    y: Optional[pd.Series] = None
    if TARGET_COLUMN in working_df.columns:
        y = working_df[TARGET_COLUMN].astype(int).copy()
        working_df = working_df.drop(columns=[TARGET_COLUMN])

    # 3. Drop excluded columns
    cols_to_drop = [c for c in EXCLUDED_COLUMNS if c in working_df.columns]
    if cols_to_drop:
        logger.info("Dropping excluded columns from feature matrix: %s", cols_to_drop)
        working_df = working_df.drop(columns=cols_to_drop)

    # 4. Ensure categorical columns are strings
    for cat_col in CATEGORICAL_COLUMNS:
        if cat_col in working_df.columns:
            working_df[cat_col] = working_df[cat_col].astype(str)

    # 5. Filter to expected model features
    expected_cols = FULL_NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    available_cols = [c for c in expected_cols if c in working_df.columns]
    X = working_df[available_cols].copy()

    logger.info(
        "Prepared feature matrix X shape: %s (Categorical: %d, Numerical: %d). Target y present: %s",
        X.shape,
        len([c for c in CATEGORICAL_COLUMNS if c in X.columns]),
        len([c for c in FULL_NUMERICAL_COLUMNS if c in X.columns]),
        y is not None,
    )
    return X, y


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct the scikit-learn ColumnTransformer preprocessing pipeline.

    Numerical Pipeline:
    - Median SimpleImputer (robust to potential missing or corrupted values)
    - StandardScaler (zero mean, unit variance for stable gradient and distance metrics)

    Categorical Pipeline:
    - Most Frequent SimpleImputer
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.

    Returns:
        ColumnTransformer: Configured preprocessor object.
    """
    num_features = numerical_cols if numerical_cols is not None else FULL_NUMERICAL_COLUMNS
    cat_features = categorical_cols if categorical_cols is not None else CATEGORICAL_COLUMNS

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_preprocessing_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Construct a unified preprocessing Pipeline combining:
    1. ClaimFeatureTransformer: Data-dependent features fitted solely on training folds.
    2. ColumnTransformer: Numerical imputation/scaling and categorical encoding.

    Returns:
        Pipeline: Preprocessing pipeline object.
    """
    column_transformer = build_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", ClaimFeatureTransformer()),
            ("column_transformer", column_transformer),
        ]
    )


def build_full_pipeline(
    model: Any,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Pipeline:
    """
    Construct an end-to-end ML pipeline with complete leakage prevention:
    [Feature Engineering -> Preprocessing -> Estimator Model]

    Ensures that feature engineering benchmarks and preprocessing scales are learned
    independently inside each training fold during cross-validation.

    Args:
        model: Scikit-learn compatible classifier.
        numerical_cols: Optional custom list of numerical columns.
        categorical_cols: Optional custom list of categorical columns.

    Returns:
        Pipeline: End-to-end executable pipeline.
    """
    prep_pipeline = build_preprocessing_pipeline(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )
    return Pipeline(
        steps=[
            ("pipeline_prep", prep_pipeline),
            ("model", model),
        ]
    )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split.

    Args:
        X: Feature matrix DataFrame.
        y: Target Series.
        test_size: Proportion allocated to testing split (default: 0.20).
        random_state: Deterministic random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    if len(X) != len(y):
        raise ValueError(f"X length ({len(X)}) must match y length ({len(y)})")

    logger.info(
        "Performing stratified train/test split (test_size=%.2f, random_state=%d)...",
        test_size,
        random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_pos_rate = float(y_train.mean()) * 100
    test_pos_rate = float(y_test.mean()) * 100

    logger.info("Split complete:")
    logger.info("  Train set: %d records, positive rate: %.2f%%", len(y_train), train_pos_rate)
    logger.info("  Test set:  %d records, positive rate: %.2f%%", len(y_test), test_pos_rate)

    return X_train, X_test, y_train, y_test


def fit_transform_data(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Fit preprocessor ONLY on X_train and transform both X_train and X_test.

    IMPORTANT: Prevents data leakage by strictly learning scaling parameters
    and categorical levels solely on training folds.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training split features.
        X_test: Testing split features.

    Returns:
        Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
            (X_train_proc, X_test_proc, fitted_preprocessor, feature_names)
    """
    logger.info("Fitting preprocessing transformations SOLELY on training data...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract clean transformed feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    logger.info(
        "Preprocessing complete. Transformed features: %d. Train shape: %s, Test shape: %s",
        len(feature_names),
        X_train_proc.shape,
        X_test_proc.shape,
    )

    return X_train_proc, X_test_proc, preprocessor, feature_names


def preprocess_claims(
    data_path: str | Path,
    output_dir: Optional[str | Path] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    High-level end-to-end preprocessing workflow.

    1. Loads dataset.
    2. Extracts feature matrix X and target y.
    3. Partitions into stratified train/test splits.
    4. Constructs preprocessor.
    5. Fits preprocessor strictly on training split and transforms both splits.
    6. Optionally persists processed datasets and preprocessor artifact.

    Args:
        data_path: Path to CSV dataset.
        output_dir: Optional directory to persist preprocessed splits and preprocessor.
        test_size: Testing partition ratio (default: 0.20).
        random_state: Random state seed for reproducibility (default: 42).

    Returns:
        Dict[str, Any]: Dictionary containing processed arrays, DataFrames, and fitted preprocessor.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    logger.info("Loading raw dataset from %s...", data_path.resolve())
    raw_df = pd.read_csv(data_path, dtype={"procedure_code": str})

    # Prepare features and separate target
    X, y = prepare_features(raw_df, perform_engineering=True)
    if y is None:
        raise ValueError("Dataset does not contain the required target column 'requires_review'.")

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Build and fit preprocessor
    num_cols = [c for c in FULL_NUMERICAL_COLUMNS if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]

    preprocessor = build_preprocessor(numerical_cols=num_cols, categorical_cols=cat_cols)
    X_train_proc, X_test_proc, fitted_preprocessor, feature_names = fit_transform_data(
        preprocessor,
        X_train,
        X_test,
    )

    # Form DataFrames with transformed feature names for analyst convenience
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

    results: Dict[str, Any] = {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_proc,
        "X_test_processed": X_test_proc,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "preprocessor": fitted_preprocessor,
        "feature_names": feature_names,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        logger.info("Persisting preprocessed datasets to %s...", out_path.resolve())
        X_train_df.to_csv(out_path / "X_train_processed.csv", index=False)
        X_test_df.to_csv(out_path / "X_test_processed.csv", index=False)
        y_train.to_csv(out_path / "y_train.csv", index=False)
        y_test.to_csv(out_path / "y_test.csv", index=False)

        # Also save raw engineered splits
        X_train.to_csv(out_path / "X_train_raw.csv", index=False)
        X_test.to_csv(out_path / "X_test_raw.csv", index=False)

        # Save feature names list
        with open(out_path / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)

        # Save preprocessor artifact
        preprocessor_file = out_path / "preprocessor.joblib"
        joblib.dump(fitted_preprocessor, preprocessor_file)
        logger.info("Saved fitted preprocessor to %s", preprocessor_file.resolve())

    return results


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="ClaimRisk AI - Preprocessing and Partitioning CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic/healthcare_claims.csv"),
        help="Path to input claims dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save preprocessed splits and preprocessor.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test dataset partition fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic stratified splitting.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint for preprocessing."""
    args = parse_arguments()
    try:
        preprocess_claims(
            data_path=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.seed,
        )
    except Exception as exc:
        logger.error("Preprocessing pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
