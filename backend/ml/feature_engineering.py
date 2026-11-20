"""
ClaimRisk AI - Feature Engineering Module.

This module implements reproducible, explainable, and production-ready feature
transformations for healthcare billing claims.

Domain Rationales for Engineered Features:
1. amount_to_benchmark_ratio:
   Ratio of billed claim amount to expected procedure benchmark base mean.
   Normalizes charges across diverse procedure scales (e.g. $28 venipuncture vs.
   $4,200 knee arthroplasty) to capture relative upcoding or fee inflation.

2. high_submission_delay:
   Binary indicator (delay > 60 days). Captures timely filing risk and documentation
   lag, common operational triggers for payer audits.

3. has_multiple_missing_fields:
   Binary indicator (missing_fields >= 2). Identifies claims with elevated administrative
   friction and documentation deficits beyond casual single-field omissions.

4. high_provider_rejection_rate:
   Binary indicator (historical_rejection_rate > 0.20). Flags claims originating
   from healthcare providers with elevated historical audit or claim denial rates.

5. high_provider_claim_frequency:
   Binary indicator (provider_claim_frequency > 300 claims/month). Flags high-volume
   billing facilities where billing automation errors and batch burst anomalies concentrate.

6. high_amount_deviation:
   Binary indicator (amount_deviation > 2.0). Highlights pricing outliers that exceed
   two standard deviations from the negotiated clinical benchmark.

7. service_month:
   Integer (1-12) extracted from service_date. Captures annual seasonality, deductible
   resets in Q1, and end-of-year billing volume surges.

8. service_day_of_week:
   Integer (0=Monday to 6=Sunday) extracted from service_date. Reflects weekly billing cycles
   and procedural cadence.

9. service_is_weekend:
   Binary indicator (service_day_of_week >= 5). Highlights elective outpatient procedures
   billed on weekends, a frequent billing anomaly indicator.

STRICT CONSTRAINTS:
- No target leakage: 'requires_review' is NEVER used during feature engineering.
- 'approved_amount' is strictly excluded from predictive feature creation.
- Preserves the input DataFrame (returns a clean copy).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.feature_engineering")

# Standard procedure code benchmark lookup (CPT/HCPCS proxies)
# Mean fee baselines matching clinical billing catalogs
PROCEDURE_BENCHMARKS: Dict[str, float] = {
    "99213": 110.0,   # Office outpatient visit, low complexity
    "99214": 165.0,   # Office outpatient visit, moderate complexity
    "99215": 225.0,   # Office outpatient visit, high complexity
    "99284": 480.0,   # Emergency department visit, high severity
    "99285": 760.0,   # Emergency department visit, immediate threat
    "71045": 95.0,    # Chest X-ray, single view
    "70450": 430.0,   # CT Head/Brain without contrast
    "80053": 65.0,    # Comprehensive metabolic panel
    "93000": 85.0,    # Electrocardiogram, routine ECG
    "36415": 28.0,    # Routine venipuncture collection
    "43239": 980.0,   # Upper endoscopy with biopsy
    "27447": 4200.0,  # Total knee arthroplasty
}

# Names of all engineered feature columns produced by this module
ENGINEERED_FEATURE_NAMES: List[str] = [
    "amount_to_benchmark_ratio",
    "high_submission_delay",
    "has_multiple_missing_fields",
    "high_provider_rejection_rate",
    "high_provider_claim_frequency",
    "high_amount_deviation",
    "service_month",
    "service_day_of_week",
    "service_is_weekend",
]


class ProcedureBenchmarkTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for computing procedure-code benchmark ratios.

    Strict Data Leakage Prevention:
    - Learns baseline benchmarks and empirical medians SOLELY during fit(X, y).
    - During transform(X), applies only pre-fitted benchmarks and training global median.
    - Fits independently in each cross-validation fold to prevent train/test contamination.
    """

    def __init__(self, catalog_benchmarks: Optional[Dict[str, float]] = None) -> None:
        self.catalog_benchmarks = catalog_benchmarks
        self.fitted_benchmarks_: Dict[str, float] = {}
        self.global_median_: float = 100.0

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "ProcedureBenchmarkTransformer":
        """
        Fit procedure benchmarks strictly on training data.

        Args:
            X: Training DataFrame containing 'claim_amount' and 'procedure_code'.
            y: Ignored (present for scikit-learn API compatibility).

        Returns:
            self: Fitted transformer.
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        catalog = (self.catalog_benchmarks or PROCEDURE_BENCHMARKS).copy()
        amounts = pd.to_numeric(df["claim_amount"], errors="coerce")
        self.global_median_ = float(amounts.median()) if not amounts.empty and amounts.median() > 0 else 100.0

        self.fitted_benchmarks_ = catalog.copy()
        if "procedure_code" in df.columns:
            codes = df["procedure_code"].astype(str)
            unseen = set(codes.unique()) - set(catalog.keys())
            for code in unseen:
                code_amts = amounts[codes == code].dropna()
                self.fitted_benchmarks_[code] = float(code_amts.median()) if not code_amts.empty else self.global_median_

        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """
        Transform X by computing amount_to_benchmark_ratio using fitted benchmarks.

        Args:
            X: Input DataFrame.

        Returns:
            pd.Series: Computed ratio series.
        """
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        codes = df["procedure_code"].astype(str)
        amounts = pd.to_numeric(df["claim_amount"], errors="coerce")

        base_means = codes.map(self.fitted_benchmarks_).fillna(self.global_median_)
        base_means = base_means.replace(0, np.nan).fillna(self.global_median_)
        ratio = (amounts / base_means).round(4)
        return ratio.rename("amount_to_benchmark_ratio")


class ClaimFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for ClaimRisk AI claims.

    Integrates:
    - Train-fitted procedure benchmark ratios (zero train/test leakage)
    - Non-linear operational risk threshold indicators
    - Temporal and calendar features from service dates
    """

    def __init__(
        self,
        catalog_benchmarks: Optional[Dict[str, float]] = None,
        delay_threshold: int = 60,
        missing_fields_threshold: int = 2,
        rejection_rate_threshold: float = 0.20,
        claim_frequency_threshold: int = 300,
        amount_deviation_threshold: float = 2.0,
    ) -> None:
        self.catalog_benchmarks = catalog_benchmarks
        self.delay_threshold = delay_threshold
        self.missing_fields_threshold = missing_fields_threshold
        self.rejection_rate_threshold = rejection_rate_threshold
        self.claim_frequency_threshold = claim_frequency_threshold
        self.amount_deviation_threshold = amount_deviation_threshold
        self.benchmark_transformer_ = ProcedureBenchmarkTransformer(catalog_benchmarks)

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "ClaimFeatureTransformer":
        """Fit transformer strictly on training split."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.benchmark_transformer_.fit(df, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input features using training-fitted transformers."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        enriched = df.copy()

        # 1. Ratio using training-fitted benchmarks
        enriched["amount_to_benchmark_ratio"] = self.benchmark_transformer_.transform(enriched)

        # 2. Threshold indicators
        indicators = extract_threshold_indicators(
            enriched,
            delay_threshold=self.delay_threshold,
            missing_fields_threshold=self.missing_fields_threshold,
            rejection_rate_threshold=self.rejection_rate_threshold,
            claim_frequency_threshold=self.claim_frequency_threshold,
            amount_deviation_threshold=self.amount_deviation_threshold,
        )
        for col in indicators.columns:
            enriched[col] = indicators[col]

        # 3. Date features
        if "service_date" in enriched.columns:
            date_feats = extract_date_features(enriched, date_col="service_date")
            for col in date_feats.columns:
                enriched[col] = date_feats[col]

        return enriched


def compute_benchmark_ratio(
    df: pd.DataFrame,
    benchmarks: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Compute ratio of billed claim_amount to normative procedure benchmark mean.

    Handles unknown procedure codes gracefully by falling back to empirical median
    claim amount for that procedure, or global median if completely unseen.

    Args:
        df: DataFrame containing 'procedure_code' and 'claim_amount'.
        benchmarks: Optional dictionary mapping procedure codes to benchmark base mean.

    Returns:
        pd.Series: Continuous ratio feature (e.g. 1.0 = exactly at benchmark, 2.5 = 2.5x benchmark).
    """
    if "claim_amount" not in df.columns or "procedure_code" not in df.columns:
        raise KeyError("DataFrame must contain 'claim_amount' and 'procedure_code'.")

    transformer = ProcedureBenchmarkTransformer(catalog_benchmarks=benchmarks)
    transformer.fit(df)
    return transformer.transform(df)


def extract_date_features(df: pd.DataFrame, date_col: str = "service_date") -> pd.DataFrame:
    """
    Extract calendar and operational temporal features from service date.

    Engineered features:
    - service_month: 1 to 12
    - service_day_of_week: 0 (Monday) to 6 (Sunday)
    - service_is_weekend: 1 if Saturday or Sunday, else 0

    Args:
        df: DataFrame containing date_col.
        date_col: Column name containing date strings or datetime objects.

    Returns:
        pd.DataFrame: 3-column DataFrame with extracted date features.
    """
    if date_col not in df.columns:
        raise KeyError(f"DataFrame missing required date column: '{date_col}'")

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

    month = parsed_dates.dt.month.fillna(1).astype(int)
    if hasattr(parsed_dates.dt, "day_of_week"):
        day_of_week = parsed_dates.dt.day_of_week.fillna(0).astype(int)
    else:
        day_of_week = parsed_dates.dt.dayofweek.fillna(0).astype(int)
    is_weekend = (day_of_week >= 5).astype(int)

    return pd.DataFrame(
        {
            "service_month": month,
            "service_day_of_week": day_of_week,
            "service_is_weekend": is_weekend,
        },
        index=df.index,
    )


def extract_threshold_indicators(
    df: pd.DataFrame,
    delay_threshold: int = 60,
    missing_fields_threshold: int = 2,
    rejection_rate_threshold: float = 0.20,
    claim_frequency_threshold: int = 300,
    amount_deviation_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Compute binary domain risk indicators from clinical, administrative, and provider baselines.

    Args:
        df: Input DataFrame containing required operational columns.
        delay_threshold: Threshold in days for high submission delay flag (default: 60).
        missing_fields_threshold: Threshold count for multiple missing fields flag (default: 2).
        rejection_rate_threshold: Historical rejection rate threshold (default: 0.20).
        claim_frequency_threshold: Monthly provider claim volume threshold (default: 300).
        amount_deviation_threshold: Standard deviations above benchmark (default: 2.0).

    Returns:
        pd.DataFrame: DataFrame containing 5 binary indicator flags.
    """
    required = [
        "submission_delay",
        "missing_fields",
        "historical_rejection_rate",
        "provider_claim_frequency",
        "amount_deviation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing columns required for indicator extraction: {missing}")

    high_delay = (df["submission_delay"] > delay_threshold).astype(int)
    has_multiple_missing = (df["missing_fields"] >= missing_fields_threshold).astype(int)
    high_rejection = (df["historical_rejection_rate"] > rejection_rate_threshold).astype(int)
    high_freq = (df["provider_claim_frequency"] > claim_frequency_threshold).astype(int)
    high_dev = (df["amount_deviation"] > amount_deviation_threshold).astype(int)

    return pd.DataFrame(
        {
            "high_submission_delay": high_delay,
            "has_multiple_missing_fields": has_multiple_missing,
            "high_provider_rejection_rate": high_rejection,
            "high_provider_claim_frequency": high_freq,
            "high_amount_deviation": high_dev,
        },
        index=df.index,
    )


def validate_engineered_features(df: pd.DataFrame) -> None:
    """
    Validate that all expected engineered features exist and adhere to domain sanity rules.

    Args:
        df: DataFrame containing newly engineered features.

    Raises:
        ValueError: If any validation checks fail.
    """
    for col in ENGINEERED_FEATURE_NAMES:
        if col not in df.columns:
            raise ValueError(f"Expected engineered feature '{col}' is missing.")

        if df[col].isnull().any():
            null_count = int(df[col].isnull().sum())
            raise ValueError(f"Feature '{col}' contains {null_count} unexpected null values.")

    # Bound validations
    if not df["service_month"].between(1, 12).all():
        raise ValueError("service_month values must strictly be between 1 and 12.")

    if not df["service_day_of_week"].between(0, 6).all():
        raise ValueError("service_day_of_week values must strictly be between 0 and 6.")

    binary_cols = [
        "high_submission_delay",
        "has_multiple_missing_fields",
        "high_provider_rejection_rate",
        "high_provider_claim_frequency",
        "high_amount_deviation",
        "service_is_weekend",
    ]
    for b_col in binary_cols:
        if not set(df[b_col].unique()).issubset({0, 1}):
            raise ValueError(f"Binary feature '{b_col}' contains invalid values: {df[b_col].unique()}")

    if (df["amount_to_benchmark_ratio"] <= 0).any():
        raise ValueError("amount_to_benchmark_ratio contains non-positive values.")


def engineer_features(
    df: pd.DataFrame,
    benchmarks: Optional[Dict[str, float]] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Transform raw healthcare claim records into an enriched feature set.

    Preserves the original DataFrame by returning a clean copy.
    Avoids using 'requires_review' or 'approved_amount' in any transformation.

    Args:
        df: Raw claims DataFrame.
        benchmarks: Optional benchmark dictionary for procedure codes.
        validate: If True, validates generated features against sanity rules.

    Returns:
        pd.DataFrame: Enriched DataFrame containing original columns plus engineered features.
    """
    if df.empty:
        logger.warning("Received empty DataFrame for feature engineering.")
        result = df.copy()
        for col in ENGINEERED_FEATURE_NAMES:
            result[col] = pd.Series(dtype=float if "ratio" in col else int)
        return result

    logger.info("Executing feature engineering pipeline on %d records...", len(df))

    # Defensive copy: do NOT mutate the original DataFrame
    enriched = df.copy()

    # 1. Benchmark ratio feature
    enriched["amount_to_benchmark_ratio"] = compute_benchmark_ratio(enriched, benchmarks=benchmarks)

    # 2. Risk threshold indicators
    indicators_df = extract_threshold_indicators(enriched)
    for col in indicators_df.columns:
        enriched[col] = indicators_df[col]

    # 3. Calendar & temporal features from service_date
    date_features_df = extract_date_features(enriched, date_col="service_date")
    for col in date_features_df.columns:
        enriched[col] = date_features_df[col]

    if validate:
        validate_engineered_features(enriched)

    logger.info(
        "Feature engineering complete. Added %d new features. Total columns: %d",
        len(ENGINEERED_FEATURE_NAMES),
        len(enriched.columns),
    )
    return enriched


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for standalone feature engineering."""
    parser = argparse.ArgumentParser(
        description="ClaimRisk AI - Feature Engineering Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic/healthcare_claims.csv"),
        help="Path to input raw CSV claims dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/claims_engineered.csv"),
        help="Path to save enriched dataset with engineered features.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint for feature engineering."""
    args = parse_arguments()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        logger.error("Input dataset file not found: %s", input_path.resolve())
        sys.exit(1)

    logger.info("Loading dataset from: %s", input_path.resolve())
    raw_df = pd.read_csv(input_path, dtype={"procedure_code": str})

    engineered_df = engineer_features(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output_path, index=False)
    logger.info("Enriched dataset successfully saved to: %s", output_path.resolve())


if __name__ == "__main__":
    main()
