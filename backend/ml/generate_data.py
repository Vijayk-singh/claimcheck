"""
ClaimRisk AI - Synthetic Healthcare Claims Generator (Stage 1)

Generates a realistic, reproducible healthcare billing claims dataset designed for
explainable risk triage, anomaly detection, and analyst copilot workflows.

Key Modeling Considerations:
- Probabilistic target assignment via latent logistic score (avoiding trivial rule leaks).
- Plausible procedure-based pricing baselines and controlled billing anomalies.
- Operational separation between pre-adjudication features and post-adjudication outcomes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.data_generator")


@dataclass(frozen=True)
class ProcedureBenchmark:
    """Benchmark cost distribution for a standard healthcare procedure code."""

    code: str
    description: str
    category: str
    base_mean: float
    base_std: float
    frequency_weight: float


# Representative clinical and billing procedure code catalog (CPT / HCPCS standard proxies)
PROCEDURE_CATALOG: Tuple[ProcedureBenchmark, ...] = (
    ProcedureBenchmark("99213", "Office outpatient visit, low complexity", "Evaluation & Management", 110.0, 15.0, 0.22),
    ProcedureBenchmark("99214", "Office outpatient visit, moderate complexity", "Evaluation & Management", 165.0, 25.0, 0.20),
    ProcedureBenchmark("99215", "Office outpatient visit, high complexity", "Evaluation & Management", 225.0, 35.0, 0.08),
    ProcedureBenchmark("99284", "Emergency department visit, high severity", "Emergency", 480.0, 65.0, 0.07),
    ProcedureBenchmark("99285", "Emergency department visit, immediate threat", "Emergency", 760.0, 110.0, 0.05),
    ProcedureBenchmark("71045", "Chest X-ray, single view", "Radiology", 95.0, 18.0, 0.10),
    ProcedureBenchmark("70450", "CT Head/Brain without contrast", "Radiology", 430.0, 60.0, 0.05),
    ProcedureBenchmark("80053", "Comprehensive metabolic panel", "Pathology/Lab", 65.0, 12.0, 0.08),
    ProcedureBenchmark("93000", "Electrocardiogram, routine ECG", "Cardiovascular", 85.0, 14.0, 0.06),
    ProcedureBenchmark("36415", "Routine venipuncture collection", "Pathology/Lab", 28.0, 5.0, 0.05),
    ProcedureBenchmark("43239", "Upper endoscopy with biopsy", "Surgery/Gastroenterology", 980.0, 130.0, 0.02),
    ProcedureBenchmark("27447", "Total knee arthroplasty", "Surgery/Orthopedics", 4200.0, 520.0, 0.02),
)


class SyntheticClaimGenerator:
    """
    Modular generator for synthetic healthcare claims with controlled billing patterns,
    correlated provider behaviors, and a probabilistic review prioritization target.
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Initialize generator with a deterministic random seed.

        Args:
            seed: Integer random seed for NumPy and random state reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.catalog_by_code: Dict[str, ProcedureBenchmark] = {p.code: p for p in PROCEDURE_CATALOG}

    def _generate_provider_population(self, num_providers: int = 250) -> pd.DataFrame:
        """
        Generate provider master profiles with inherent behavioral baselines:
        - 80% standard compliant providers
        - 12% high-volume batch billing facilities
        - 8% elevated audit history / billing anomaly providers
        """
        provider_ids = [f"PRV-{i+1:04d}" for i in range(num_providers)]

        # Assign provider profile tiers
        tiers = self.rng.choice(
            ["standard", "high_volume", "elevated_risk"],
            size=num_providers,
            p=[0.80, 0.12, 0.08],
        )

        monthly_frequencies = []
        historical_rejection_rates = []

        for tier in tiers:
            if tier == "standard":
                freq = int(self.rng.gamma(shape=5.0, scale=12.0)) + 20  # ~80 claims/mo
                rej = float(self.rng.beta(a=2.0, b=28.0))             # ~6.6% rejection
            elif tier == "high_volume":
                freq = int(self.rng.gamma(shape=12.0, scale=25.0)) + 150  # ~450 claims/mo
                rej = float(self.rng.beta(a=3.5, b=22.0))              # ~13.7% rejection
            else:  # elevated_risk
                freq = int(self.rng.gamma(shape=6.0, scale=18.0)) + 40   # ~150 claims/mo
                rej = float(self.rng.beta(a=6.0, b=14.0))              # ~30.0% rejection

            monthly_frequencies.append(max(10, min(freq, 800)))
            historical_rejection_rates.append(round(float(np.clip(rej, 0.01, 0.65)), 4))

        return pd.DataFrame(
            {
                "provider_id": provider_ids,
                "provider_tier": tiers,
                "provider_claim_frequency": monthly_frequencies,
                "historical_rejection_rate": historical_rejection_rates,
            }
        )

    def generate_dataset(self, num_records: int = 20000) -> pd.DataFrame:
        """
        Generate synthetic healthcare billing claims dataset.

        Args:
            num_records: Number of claim records to generate (default: 20,000).

        Returns:
            pd.DataFrame: Validated synthetic claims dataframe.
        """
        if num_records <= 0:
            raise ValueError(f"num_records must be a positive integer, received: {num_records}")

        logger.info("Starting generation of %d claims with seed=%d...", num_records, self.seed)

        # 1. Providers Population & Assignment
        num_providers = max(50, num_records // 80)
        providers_df = self._generate_provider_population(num_providers=num_providers)

        # Sample providers with weighting proportional to their monthly frequency
        freq_weights = providers_df["provider_claim_frequency"].to_numpy(dtype=float)
        provider_probs = freq_weights / freq_weights.sum()

        chosen_provider_indices = self.rng.choice(
            len(providers_df), size=num_records, p=provider_probs
        )
        sampled_providers = providers_df.iloc[chosen_provider_indices].reset_index(drop=True)

        # 2. Procedure Code Sampling
        codes = [p.code for p in PROCEDURE_CATALOG]
        code_weights = np.array([p.frequency_weight for p in PROCEDURE_CATALOG], dtype=float)
        code_probs = code_weights / code_weights.sum()

        chosen_codes = self.rng.choice(codes, size=num_records, p=code_probs)

        # 3. Dates & Submission Delay
        # Service date spanning 18 months
        start_date = date(2023, 1, 1)
        max_days = 540
        service_offsets = self.rng.integers(0, max_days, size=num_records)
        service_dates = [start_date + timedelta(days=int(offset)) for offset in service_offsets]

        # Submission delay (days): majority within 5-35 days, small fraction delayed
        # Mixture model: 93% standard delay, 7% delayed billing / timely filing risk
        is_delayed_submission = self.rng.binomial(1, 0.07, size=num_records)
        standard_delays = self.rng.gamma(shape=3.5, scale=4.0, size=num_records) + 2.0  # mean ~16 days
        lagged_delays = self.rng.uniform(60, 210, size=num_records)                    # 60-210 days

        submission_delays = np.where(
            is_delayed_submission == 1,
            lagged_delays,
            standard_delays,
        ).astype(int)
        submission_delays = np.clip(submission_delays, 1, 365)

        submission_dates = [
            s_date + timedelta(days=int(delay))
            for s_date, delay in zip(service_dates, submission_delays)
        ]

        # 4. Missing Fields Count
        # Counts of missing non-critical claim fields (0 to 4)
        missing_fields = self.rng.choice(
            [0, 1, 2, 3, 4],
            size=num_records,
            p=[0.68, 0.20, 0.08, 0.03, 0.01],
        )

        # 5. Duplicate Submission Indicator
        # Healthcare payers often encounter duplicate re-submissions (~2.5% baseline)
        # Higher likelihood in high-volume or elevated-risk providers
        is_elevated_provider = (sampled_providers["provider_tier"] == "elevated_risk").to_numpy()
        dup_probs = np.where(is_elevated_provider, 0.07, 0.02)
        duplicate_indicator = self.rng.binomial(1, dup_probs, size=num_records)

        # 6. Claim Amount & Amount Deviation
        # Benchmark-informed pricing with controlled anomaly injection (upcoding, fee skew)
        claim_amounts = np.zeros(num_records, dtype=float)
        amount_deviations = np.zeros(num_records, dtype=float)

        # Anomaly rate: ~6% inflated charges / upcoding patterns
        is_inflated = self.rng.binomial(1, 0.06, size=num_records)

        for i, code in enumerate(chosen_codes):
            benchmark = self.catalog_by_code[code]
            b_mean = benchmark.base_mean
            b_std = benchmark.base_std

            if is_inflated[i] == 1:
                # Upcoding / anomalous charge: 2.2x to 5.5x base mean
                multiplier = self.rng.uniform(2.2, 5.5)
                amt = b_mean * multiplier + self.rng.normal(0, b_std)
            else:
                # Normal variation around negotiated benchmark
                amt = self.rng.normal(b_mean, b_std)

            # Ensure minimum viable charge floor (at least 30% of benchmark)
            amt = max(amt, b_mean * 0.3)
            z_dev = (amt - b_mean) / b_std

            claim_amounts[i] = round(amt, 2)
            amount_deviations[i] = round(z_dev, 3)

        # 7. Probabilistic Target Assignment: requires_review
        # Derived from a multi-factor latent logit model + stochastic noise
        # Guarantees that no single feature deterministically dictates the outcome.
        target_labels, target_probs = self._compute_review_target(
            amount_deviation=amount_deviations,
            duplicate_indicator=duplicate_indicator,
            historical_rejection_rate=sampled_providers["historical_rejection_rate"].to_numpy(),
            submission_delay=submission_delays,
            missing_fields=missing_fields,
            is_elevated_provider=is_elevated_provider,
        )

        # 8. Approved Amount (Post-Adjudication Outcome)
        # Note for ML engineers: In live workflows, approved_amount is established
        # AFTER review/adjudication. We generate it here to complete the claims lifecycle
        # record, but it must be excluded from pre-adjudication predictive features.
        approved_amounts = self._simulate_approved_amount(
            claim_amounts=claim_amounts,
            chosen_codes=chosen_codes,
            requires_review=target_labels,
            duplicate_indicator=duplicate_indicator,
        )

        # 9. Format Final DataFrame matching required schema
        claim_ids = [f"CLM-{self.seed}-{i+1:06d}" for i in range(num_records)]

        df = pd.DataFrame(
            {
                "claim_id": claim_ids,
                "provider_id": sampled_providers["provider_id"],
                "procedure_code": chosen_codes,
                "claim_amount": claim_amounts,
                "approved_amount": approved_amounts,
                "service_date": [d.strftime("%Y-%m-%d") for d in service_dates],
                "submission_date": [d.strftime("%Y-%m-%d") for d in submission_dates],
                "submission_delay": submission_delays,
                "missing_fields": missing_fields,
                "duplicate_indicator": duplicate_indicator,
                "provider_claim_frequency": sampled_providers["provider_claim_frequency"],
                "historical_rejection_rate": sampled_providers["historical_rejection_rate"],
                "amount_deviation": amount_deviations,
                "requires_review": target_labels,
            }
        )

        logger.info(
            "Dataset generation complete: %d records, %d features. Review rate: %.2f%%",
            len(df),
            len(df.columns),
            (df["requires_review"].mean() * 100),
        )

        return df

    def _compute_review_target(
        self,
        amount_deviation: np.ndarray,
        duplicate_indicator: np.ndarray,
        historical_rejection_rate: np.ndarray,
        submission_delay: np.ndarray,
        missing_fields: np.ndarray,
        is_elevated_provider: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute probabilistic triage target using latent logistic scoring.

        Interactions modeled:
        - High amount deviation with elevated historical provider rejections.
        - Duplicate indicator with administrative delays.
        - Missing documentation friction.
        """
        # Feature standardization for logit calculation
        z_amt = np.clip(amount_deviation, -2.0, 6.0)
        z_dup = duplicate_indicator.astype(float)
        z_rej = (historical_rejection_rate - 0.10) / 0.10
        z_delay = np.clip((submission_delay - 15) / 30.0, -0.5, 4.0)
        z_missing = missing_fields.astype(float)
        z_interaction = np.maximum(0.0, z_amt) * np.maximum(0.0, z_rej)

        # Baseline logit calibrated to ~18% base review rate
        # Intercept (-2.2) + weighted risk factors + Gaussian noise
        latent_logit = (
            -2.20
            + 0.90 * z_amt
            + 1.85 * z_dup
            + 0.80 * z_rej
            + 0.45 * z_delay
            + 0.55 * z_missing
            + 0.35 * z_interaction
            + 0.50 * is_elevated_provider.astype(float)
        )

        # Controlled stochastic noise representing unobserved auditor factors & human variance
        noise = self.rng.normal(0.0, 0.45, size=len(amount_deviation))
        noisy_logit = latent_logit + noise

        # Logistic sigmoid transformation
        review_probs = 1.0 / (1.0 + np.exp(-noisy_logit))

        # Bernoulli trial for final label
        target_labels = self.rng.binomial(1, review_probs)

        return target_labels, review_probs

    def _simulate_approved_amount(
        self,
        claim_amounts: np.ndarray,
        chosen_codes: List[str],
        requires_review: np.ndarray,
        duplicate_indicator: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate contractual adjudication payment (approved_amount).

        In clean approved claims, payers pay contractual rates (typically 80-95% of benchmark).
        Claims flagged for review with duplicates or gross deviations face payment denial ($0)
        or downward adjustment pending review resolution.
        """
        approved = np.zeros(len(claim_amounts), dtype=float)

        for i, code in enumerate(chosen_codes):
            benchmark = self.catalog_by_code[code]
            billed = claim_amounts[i]
            needs_review = requires_review[i]
            is_dup = duplicate_indicator[i]

            if is_dup == 1:
                # Denied duplicate submission pending audit
                approved[i] = 0.0
            elif needs_review == 1:
                # If flagged for review, payment is either withheld ($0) or capped at benchmark allowable
                if self.rng.random() < 0.45:
                    approved[i] = 0.0
                else:
                    allowable = benchmark.base_mean * self.rng.uniform(0.75, 0.90)
                    approved[i] = min(billed, allowable)
            else:
                # Standard clean claim adjudication: ~85-95% of allowable/billed
                allowable = min(billed, benchmark.base_mean * 1.05)
                approved[i] = allowable * self.rng.uniform(0.85, 0.96)

            approved[i] = round(max(0.0, approved[i]), 2)

        return approved


def validate_claims_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the generated dataset against business rules, schema constraints, and sanity checks.

    Args:
        df: Pandas DataFrame containing generated claims.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_or_warning_messages)
    """
    issues: List[str] = []

    required_columns = [
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

    # 1. Schema check
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return False, issues

    # 2. Null values check
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        issues.append(f"Found unexpected null values in columns: {cols_with_nulls}")

    # 3. Primary key uniqueness
    if df["claim_id"].duplicated().any():
        dups = df["claim_id"].duplicated().sum()
        issues.append(f"claim_id contains {dups} duplicate entries. Must be strictly unique.")

    # 4. Numeric boundary constraints
    if (df["claim_amount"] <= 0).any():
        issues.append("claim_amount contains non-positive values.")

    if (df["approved_amount"] < 0).any():
        issues.append("approved_amount contains negative values.")

    if (df["approved_amount"] > df["claim_amount"]).any():
        num_exceed = (df["approved_amount"] > df["claim_amount"]).sum()
        issues.append(f"approved_amount exceeds billed claim_amount in {num_exceed} records.")

    if (df["submission_delay"] < 0).any():
        issues.append("submission_delay contains negative delay values.")

    if not df["duplicate_indicator"].isin([0, 1]).all():
        issues.append("duplicate_indicator must only contain 0 or 1.")

    if not df["requires_review"].isin([0, 1]).all():
        issues.append("requires_review target must only contain 0 or 1.")

    if ((df["historical_rejection_rate"] < 0.0) | (df["historical_rejection_rate"] > 1.0)).any():
        issues.append("historical_rejection_rate values must be within [0.0, 1.0].")

    # 5. Date consistency
    s_dates = pd.to_datetime(df["service_date"], errors="coerce")
    sub_dates = pd.to_datetime(df["submission_date"], errors="coerce")
    if s_dates.isnull().any() or sub_dates.isnull().any():
        issues.append("Date parsing failure detected in service_date or submission_date.")
    else:
        computed_delay = (sub_dates - s_dates).dt.days
        if not (computed_delay == df["submission_delay"]).all():
            mismatch_count = (computed_delay != df["submission_delay"]).sum()
            issues.append(f"submission_delay does not match date difference in {mismatch_count} rows.")

    # 6. Target distribution sanity check (review rate should be plausible, e.g., 10% - 35%)
    review_rate = df["requires_review"].mean()
    if review_rate < 0.05 or review_rate > 0.45:
        issues.append(f"Warning: Unusual review_rate {review_rate:.2%}, expected between 10% and 35%.")

    is_valid = len(issues) == 0
    return is_valid, issues


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the synthetic claims generator."""
    parser = argparse.ArgumentParser(
        description="ClaimRisk AI - Synthetic Healthcare Claims Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=20000,
        help="Number of claim records to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/healthcare_claims.csv"),
        help="Output CSV file path.",
    )
    return parser.parse_args()


def main() -> None:
    """Execution entry point for dataset generation CLI."""
    args = parse_arguments()

    try:
        generator = SyntheticClaimGenerator(seed=args.seed)
        claims_df = generator.generate_dataset(num_records=args.size)

        # Validate dataset
        is_valid, validation_issues = validate_claims_dataset(claims_df)
        if not is_valid:
            logger.error("Dataset validation failed with errors:")
            for issue in validation_issues:
                logger.error("  - %s", issue)
            sys.exit(1)

        # Ensure parent output directory exists
        output_path: Path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        claims_df.to_csv(output_path, index=False)
        logger.info("Successfully exported %d records to %s", len(claims_df), output_path)

        # Print validation summary
        print("\n" + "=" * 60)
        print(" CLAIMRISK AI - DATASET GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total Records Generated:      {len(claims_df):,}")
        print(f"Total Columns:                {len(claims_df.columns)}")
        print(f"Output File:                  {output_path.resolve()}")
        print(f"Target Distribution (Review): {claims_df['requires_review'].value_counts().to_dict()} "
              f"({claims_df['requires_review'].mean():.2%} positive)")
        print(f"Unique Providers:             {claims_df['provider_id'].nunique()}")
        print(f"Unique Procedure Codes:       {claims_df['procedure_code'].nunique()}")
        print(f"Mean Claim Amount:            ${claims_df['claim_amount'].mean():.2f}")
        print(f"Mean Approved Amount:         ${claims_df['approved_amount'].mean():.2f}")
        print(f"Duplicate Claims Flagged:     {claims_df['duplicate_indicator'].sum():,} "
              f"({claims_df['duplicate_indicator'].mean():.2%})")
        print("Dataset Schema Validation:    PASSED [OK]")
        print("=" * 60 + "\n")

    except Exception as exc:
        logger.exception("Fatal error during dataset generation: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
