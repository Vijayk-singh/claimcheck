"""
ClaimRisk AI - Claims Dataset Validation Tool.

Validates any generated CSV claims dataset against the official schema,
statistical sanity bounds, integrity constraints, and domain business rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backend.ml.generate_data import validate_claims_dataset


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for dataset validation."""
    parser = argparse.ArgumentParser(
        description="ClaimRisk AI - Healthcare Claims Dataset Validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic/healthcare_claims.csv"),
        help="Path to CSV dataset to validate.",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entry point for validation."""
    args = parse_arguments()
    csv_path: Path = args.input

    if not csv_path.exists():
        print(f"[ERROR] Dataset file not found: {csv_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {csv_path.resolve()} ...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV file: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df):,} records with {len(df.columns)} columns.")
    is_valid, issues = validate_claims_dataset(df)

    print("\n" + "=" * 60)
    print(" CLAIMRISK AI - DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Target distribution: {df['requires_review'].value_counts().to_dict()}")
    print(f"Positive review rate: {df['requires_review'].mean():.2%}")
    print(f"Null count across all columns: {df.isnull().sum().sum()}")
    print(f"Primary key uniqueness (claim_id): {df['claim_id'].nunique() == len(df)}")

    if is_valid:
        print("\nSTATUS: [VALID] Dataset satisfies all schema and integrity checks.")
        print("=" * 60 + "\n")
        sys.exit(0)
    else:
        print("\nSTATUS: [INVALID] Validation issues discovered:")
        for idx, issue in enumerate(issues, 1):
            print(f"  {idx}. {issue}")
        print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
