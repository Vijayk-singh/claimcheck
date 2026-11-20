"""
Tests for Explainability & SHAP Module.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.ml.explain import ClaimRiskExplainer


def test_explainer_local_instance() -> None:
    """Verify local feature attribution for a single record."""
    np.random.seed(42)
    feature_names = [f"feat_{i}" for i in range(10)]
    X = np.random.randn(50, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)

    explainer = ClaimRiskExplainer(model=clf, feature_names=feature_names, background_data=X[:10])

    inst = X[0]
    local_exp = explainer.explain_instance(inst, top_k=3)

    assert "predicted_risk_probability" in local_exp
    assert "top_risk_drivers" in local_exp
    assert "top_mitigating_factors" in local_exp
    assert "all_attributions" in local_exp
    assert len(local_exp["all_attributions"]) == 10


def test_explainer_global_importance() -> None:
    """Verify global importance ranking across sample records."""
    np.random.seed(42)
    feature_names = [f"feat_{i}" for i in range(8)]
    X = np.random.randn(30, 8)
    y = (X[:, 0] > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)

    explainer = ClaimRiskExplainer(model=clf, feature_names=feature_names, background_data=X[:10])
    ranking = explainer.compute_global_importance(X[:15], sample_size=15)

    assert len(ranking) == 8
    assert all("mean_abs_shap" in r for r in ranking)
    assert ranking[0]["mean_abs_shap"] >= ranking[-1]["mean_abs_shap"]
