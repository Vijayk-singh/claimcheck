"""
ClaimRisk AI - Prediction & Review Triage Pipeline.

Provides high-throughput inference, risk scoring, tier assignment, and review prioritization:
- Pre-adjudication single-record scoring and batch scoring.
- Feature engineering & ColumnTransformer preprocessing pipeline execution.
- Risk Tier classification: LOW (<0.30), MEDIUM (0.30 - 0.70), HIGH (>=0.70).
- Financial exposure prioritization (risk_score * claim_amount).
- Local SHAP feature attribution integration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.explain import ClaimRiskExplainer
from backend.ml.feature_engineering import engineer_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.predict")

# Risk tier thresholds
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70


class ClaimRiskScorer:
    """End-to-end inference engine for ClaimRisk AI."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        feature_names_path: Optional[Union[str, Path]] = None,
        explainer_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_path = Path(model_path or PROJECT_ROOT / "models" / "champion_model.joblib")
        self.preprocessor_path = Path(
            preprocessor_path or PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib"
        )
        self.feature_names_path = Path(
            feature_names_path or PROJECT_ROOT / "data" / "processed" / "feature_names.json"
        )
        self.explainer_path = Path(
            explainer_path or PROJECT_ROOT / "reports" / "explainability" / "explainer.joblib"
        )

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model, fitted preprocessor, and feature schema."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. Run training pipeline first."
            )
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}. Run preprocessing first."
            )

        logger.info(f"Loading champion model from {self.model_path}...")
        self.model = joblib.load(self.model_path)

        logger.info(f"Loading fitted preprocessor from {self.preprocessor_path}...")
        self.preprocessor = joblib.load(self.preprocessor_path)

        if self.feature_names_path.exists():
            with open(self.feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = []

        # Load or initialize explainer
        if self.explainer_path.exists():
            try:
                self.explainer = joblib.load(self.explainer_path)
            except Exception:
                self.explainer = ClaimRiskExplainer(self.model, self.feature_names)
        else:
            self.explainer = ClaimRiskExplainer(self.model, self.feature_names)

    def _preprocess_records(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Apply feature engineering and fitted ColumnTransformer to raw input dataframe."""
        df_copy = df.copy()

        # Ensure required base columns exist with sensible defaults if omitted
        defaults = {
            "claim_id": "CLM-UNKNOWN",
            "provider_id": "PRV-UNKNOWN",
            "procedure_code": "99214",
            "claim_amount": 150.0,
            "service_date": "2024-03-01",
            "submission_date": "2024-03-15",
            "submission_delay": 14,
            "missing_fields": 0,
            "duplicate_indicator": 0,
            "provider_claim_frequency": 150,
            "historical_rejection_rate": 0.08,
            "amount_deviation": 0.0,
        }
        for col, default_val in defaults.items():
            if col not in df_copy.columns:
                df_copy[col] = default_val

        # 1. Feature Engineering (adds date features, ratio benchmarks, threshold flags)
        df_engineered = engineer_features(df_copy)

        # 2. Transform using pre-fitted ColumnTransformer
        X_processed = self.preprocessor.transform(df_engineered)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        return np.asarray(X_processed, dtype=np.float32), df_engineered

    @staticmethod
    def _determine_risk_tier(probability: float) -> str:
        """Map risk probability to operational tier."""
        if probability < LOW_THRESHOLD:
            return "LOW"
        elif probability < HIGH_THRESHOLD:
            return "MEDIUM"
        else:
            return "HIGH"

    @staticmethod
    def _recommended_action(risk_tier: str, top_driver: Optional[str] = None) -> str:
        """Suggest actionable audit next step based on risk tier and dominant risk driver."""
        if risk_tier == "LOW":
            return "Fast-track for automated adjudication. No manual audit necessary."
        elif risk_tier == "MEDIUM":
            if top_driver and "duplicate" in top_driver.lower():
                return "Perform duplicate billing cross-check against recent 90-day claims history."
            elif top_driver and "delay" in top_driver.lower():
                return "Verify timely filing documentation and proof of prior timely submission."
            return "Secondary rule validation. Verify non-critical missing documentation fields."
        else:  # HIGH
            if top_driver and "amount" in top_driver.lower():
                return "Priority clinical audit: Request itemized operative notes & verify benchmark fee markup."
            elif top_driver and "duplicate" in top_driver.lower():
                return "Immediate hold: Review suspected duplicate encounter & cross-reference billing NPI."
            elif top_driver and "rejection" in top_driver.lower():
                return "Comprehensive provider audit: High historical rejection spike detected."
            return "Priority manual review: Expedite to senior clinical adjudicator queue."

    def predict_single(
        self,
        claim_dict: Dict[str, Any],
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        """Score an individual claim and return risk level, score, and SHAP drivers."""
        df = pd.DataFrame([claim_dict])
        X_proc, df_eng = self._preprocess_records(df)

        prob = float(self.model.predict_proba(X_proc)[0, 1])
        tier = self._determine_risk_tier(prob)

        claim_amt = float(claim_dict.get("claim_amount", 0.0))
        expected_exposure = round(prob * claim_amt, 2)

        explanation: Optional[Dict[str, Any]] = None
        top_driver_name: Optional[str] = None

        if include_explanation and self.explainer is not None:
            explanation = self.explainer.explain_instance(X_proc[0])
            if explanation["top_risk_drivers"]:
                top_driver_name = explanation["top_risk_drivers"][0]["display_name"]

        action = self._recommended_action(tier, top_driver_name)

        return {
            "claim_id": str(claim_dict.get("claim_id", "CLM-UNKNOWN")),
            "provider_id": str(claim_dict.get("provider_id", "PRV-UNKNOWN")),
            "procedure_code": str(claim_dict.get("procedure_code", "UNKNOWN")),
            "claim_amount": claim_amt,
            "risk_score": round(prob, 4),
            "risk_tier": tier,
            "requires_review": int(prob >= 0.50),
            "expected_financial_exposure": expected_exposure,
            "recommended_action": action,
            "explanation": explanation,
        }

    def predict_batch(
        self,
        claims_df: pd.DataFrame,
        prioritize_by: str = "risk_score",  # 'risk_score' or 'financial_exposure'
    ) -> pd.DataFrame:
        """Batch score claims dataframe and sort triage queue for auditor efficiency."""
        df_copy = claims_df.copy()
        X_proc, df_eng = self._preprocess_records(df_copy)

        probs = self.model.predict_proba(X_proc)[:, 1]
        tiers = [self._determine_risk_tier(p) for p in probs]

        df_copy["risk_score"] = np.round(probs, 4)
        df_copy["risk_tier"] = tiers
        df_copy["requires_review"] = (probs >= 0.50).astype(int)
        df_copy["financial_exposure"] = np.round(
            probs * df_copy["claim_amount"].fillna(0.0).values, 2
        )

        if prioritize_by == "financial_exposure":
            df_copy = df_copy.sort_values(by="financial_exposure", ascending=False)
        else:
            df_copy = df_copy.sort_values(by="risk_score", ascending=False)

        return df_copy.reset_index(drop=True)


def predict_claim_risk(claim_record: dict) -> dict:
    """Convenience functional wrapper for single claim scoring."""
    scorer = ClaimRiskScorer()
    return scorer.predict_single(claim_record)


def main() -> None:
    """CLI testing tool for scoring claims."""
    parser = argparse.ArgumentParser(description="ClaimRisk AI - Claim Prediction CLI")
    parser.add_argument("--input", type=Path, help="CSV path with claims to score in batch")
    parser.add_argument("--output", type=Path, help="CSV path to save scored claims")
    parser.add_argument(
        "--prioritize",
        choices=["risk_score", "financial_exposure"],
        default="risk_score",
        help="Sorting criteria for review prioritization queue",
    )

    args = parser.parse_args()
    if args.input:
        df = pd.read_csv(args.input)
        scorer = ClaimRiskScorer()
        scored_df = scorer.predict_batch(df, prioritize_by=args.prioritize)
        if args.output:
            scored_df.to_csv(args.output, index=False)
            logger.info(f"Saved scored review queue ({len(scored_df)} claims) to {args.output}")
        else:
            print(scored_df[["claim_id", "risk_score", "risk_tier", "financial_exposure"]].head(10))
    else:
        # Smoke test with single synthetic record
        sample_claim = {
            "claim_id": "CLM-DEMO-001",
            "provider_id": "PRV-4012",
            "procedure_code": "27447",
            "claim_amount": 9500.0,
            "service_date": "2024-01-10",
            "submission_date": "2024-03-25",
            "submission_delay": 75,
            "missing_fields": 2,
            "duplicate_indicator": 1,
            "provider_claim_frequency": 350,
            "historical_rejection_rate": 0.28,
            "amount_deviation": 2.45,
        }
        scorer = ClaimRiskScorer()
        res = scorer.predict_single(sample_claim)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
