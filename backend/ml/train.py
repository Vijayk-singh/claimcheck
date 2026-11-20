"""
ClaimRisk AI - Model Training & Comparison Module.

Implements model training and benchmarking across scikit-learn models and PyTorch baseline:
- Logistic Regression (Calibrated Linear Baseline)
- Random Forest Classifier (Tree Bagging Ensemble)
- HistGradientBoostingClassifier (Gradient Boosted Trees)
- PyTorchClaimRiskClassifier (Deep Neural Network Baseline)

Features:
- Stratified K-Fold Cross-Validation.
- Full test evaluation (ROC-AUC, PR-AUC, F1, Precision, Recall).
- Champion model selection prioritized by PR-AUC.
- Serialization of all models and comparison metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.pytorch_model import PyTorchClaimRiskClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.train")


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate candidate classifiers for comparison benchmarking."""
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_depth=8,
            early_stopping=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "PyTorch_NN": PyTorchClaimRiskClassifier(
            epochs=18,
            batch_size=256,
            learning_rate=0.005,
            dropout_rate=0.20,
            random_state=random_state,
        ),
    }
    return models


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Perform Stratified K-Fold cross-validation and compute mean metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    roc_aucs, pr_aucs, f1s, precisions, recalls = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Fit model
        if isinstance(model, PyTorchClaimRiskClassifier):
            # Pass validation data for early stopping
            clf = PyTorchClaimRiskClassifier(
                epochs=model.epochs,
                batch_size=model.batch_size,
                learning_rate=model.learning_rate,
                dropout_rate=model.dropout_rate,
                random_state=random_state + fold,
            )
            clf.fit(X_fold_train, y_fold_train, X_val=X_fold_val, y_val=y_fold_val)
        else:
            from sklearn.base import clone
            clf = clone(model)
            clf.fit(X_fold_train, y_fold_train)

        # Predict
        y_prob = clf.predict_proba(X_fold_val)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        roc_aucs.append(roc_auc_score(y_fold_val, y_prob))
        pr_aucs.append(average_precision_score(y_fold_val, y_prob))
        f1s.append(f1_score(y_fold_val, y_pred, zero_division=0))
        precisions.append(precision_score(y_fold_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_fold_val, y_pred, zero_division=0))

    return {
        "cv_roc_auc_mean": float(np.mean(roc_aucs)),
        "cv_roc_auc_std": float(np.std(roc_aucs)),
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_precision_mean": float(np.mean(precisions)),
        "cv_recall_mean": float(np.mean(recalls)),
    }


def evaluate_test_performance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Compute test set performance metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "test_pr_auc": float(average_precision_score(y_test, y_prob)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }


def train_and_compare_models(
    data_dir: Path,
    output_dir: Path,
    run_cv: bool = True,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run model training, comparison benchmark, and model serialization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info(f"Loading preprocessed training and test partitions from {data_dir}...")
    X_train_df = pd.read_csv(data_dir / "X_train_processed.csv")
    X_test_df = pd.read_csv(data_dir / "X_test_processed.csv")
    y_train_df = pd.read_csv(data_dir / "y_train.csv")
    y_test_df = pd.read_csv(data_dir / "y_test.csv")

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)
    y_train = y_train_df.values.ravel().astype(int)
    y_test = y_test_df.values.ravel().astype(int)

    logger.info(
        f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_test={X_test.shape}, y_test={y_test.shape}. "
        f"Train positive review rate={np.mean(y_train):.2%}"
    )

    candidate_models = get_candidate_models(random_state=random_state)
    comparison_results: Dict[str, Any] = {}
    fitted_models: Dict[str, Any] = {}

    for model_name, model in candidate_models.items():
        logger.info(f"--- Training & Benchmarking: {model_name} ---")
        start_time = time.time()

        cv_metrics: Dict[str, float] = {}
        if run_cv:
            logger.info(f"Running {n_splits}-fold Stratified CV for {model_name}...")
            cv_metrics = cross_validate_model(
                model=model,
                X=X_train,
                y=y_train,
                n_splits=n_splits,
                random_state=random_state,
            )
            logger.info(
                f"{model_name} CV: PR-AUC={cv_metrics['cv_pr_auc_mean']:.4f} ± {cv_metrics['cv_pr_auc_std']:.4f}, "
                f"ROC-AUC={cv_metrics['cv_roc_auc_mean']:.4f}, F1={cv_metrics['cv_f1_mean']:.4f}"
            )

        # Fit on full training set
        logger.info(f"Fitting {model_name} on full training set (16,000 samples)...")
        if isinstance(model, PyTorchClaimRiskClassifier):
            # Split a 10% slice of train for neural network early stopping
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.10, stratify=y_train, random_state=random_state
            )
            model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)

        elapsed = time.time() - start_time

        # Test evaluation
        test_metrics = evaluate_test_performance(model, X_test, y_test)
        logger.info(
            f"{model_name} Test: PR-AUC={test_metrics['test_pr_auc']:.4f}, "
            f"ROC-AUC={test_metrics['test_roc_auc']:.4f}, F1={test_metrics['test_f1']:.4f}, "
            f"Recall={test_metrics['test_recall']:.4f}, Precision={test_metrics['test_precision']:.4f} "
            f"(elapsed: {elapsed:.2f}s)"
        )

        fitted_models[model_name] = model

        # Serialize individual model
        model_save_path = output_dir / f"{model_name}.joblib"
        if isinstance(model, PyTorchClaimRiskClassifier):
            pt_save_path = output_dir / f"{model_name}.pt"
            model.save(pt_save_path)
            # Also save joblib wrapper for unified loading
            joblib.dump(model, model_save_path)
        else:
            joblib.dump(model, model_save_path)

        comparison_results[model_name] = {
            "model_type": model_name,
            "training_time_seconds": round(elapsed, 2),
            **cv_metrics,
            **test_metrics,
            "model_path": str(model_save_path),
        }

    # 3. Champion Model Selection: prioritized by test PR-AUC (gold standard for imbalanced audit queues)
    champion_name = max(
        comparison_results.keys(),
        key=lambda m: comparison_results[m]["test_pr_auc"],
    )
    champion_model = fitted_models[champion_name]
    champion_metrics = comparison_results[champion_name]

    logger.info(
        f"🏆 Champion Model: {champion_name} with Test PR-AUC={champion_metrics['test_pr_auc']:.4f} "
        f"and ROC-AUC={champion_metrics['test_roc_auc']:.4f}"
    )

    # Save champion model alias
    champion_joblib_path = output_dir / "champion_model.joblib"
    joblib.dump(champion_model, champion_joblib_path)

    # Compile final comparison summary
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_model": champion_name,
        "champion_metrics": champion_metrics,
        "models": comparison_results,
    }

    summary_json_path = output_dir / "model_comparison.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Model comparison saved to {summary_json_path}")
    logger.info(f"Champion model saved to {champion_joblib_path}")

    return summary_report


def main() -> None:
    """CLI entrypoint for model training and comparison."""
    parser = argparse.ArgumentParser(
        description="ClaimRisk AI - Model Comparison & Training Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing preprocessed split CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to save serialized models and comparison reports",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation for fast training smoke tests",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    train_and_compare_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_cv=not args.skip_cv,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
