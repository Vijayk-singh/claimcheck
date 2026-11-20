"""
Tests for Model Training & Comparison Workflow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.ml.train import (
    cross_validate_model,
    evaluate_test_performance,
    get_candidate_models,
)


def test_candidate_models_instantiation() -> None:
    """Verify that all four candidate models instantiate properly."""
    models = get_candidate_models(random_state=42)
    expected_names = {"LogisticRegression", "RandomForest", "HistGradientBoosting", "PyTorch_NN"}
    assert set(models.keys()) == expected_names


def test_cross_validate_model_smoke() -> None:
    """Smoke test for Stratified K-Fold cross-validation calculation."""
    np.random.seed(42)
    X = np.random.randn(90, 10).astype(np.float32)
    y = np.random.choice([0, 1], size=90, p=[0.70, 0.30])

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    cv_res = cross_validate_model(model, X, y, n_splits=3, random_state=42)

    assert "cv_pr_auc_mean" in cv_res
    assert "cv_roc_auc_mean" in cv_res
    assert "cv_f1_mean" in cv_res
    assert 0.0 <= cv_res["cv_pr_auc_mean"] <= 1.0


def test_evaluate_test_performance() -> None:
    """Verify test metrics calculation."""
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = np.random.randn(20, 5)
    y_test = (X_test[:, 0] > 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    metrics = evaluate_test_performance(clf, X_test, y_test, threshold=0.50)
    assert "test_pr_auc" in metrics
    assert "test_roc_auc" in metrics
    assert "test_f1" in metrics
    assert "test_recall" in metrics
    assert "test_precision" in metrics
