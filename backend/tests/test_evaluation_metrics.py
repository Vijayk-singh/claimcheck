"""
Tests for Model Evaluation & Threshold Analysis Module.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.ml.evaluate import compute_threshold_metrics


def test_compute_threshold_metrics_sweep() -> None:
    """Verify that threshold analysis sweeps metrics and computes optimal thresholds."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.2, 0.85])

    res = compute_threshold_metrics(
        y_true=y_true,
        y_prob=y_prob,
        cost_audit=25.0,
        cost_missed_risk=150.0,
        thresholds=np.linspace(0.1, 0.9, 9),
    )

    assert "thresholds" in res
    assert "precision" in res
    assert "recall" in res
    assert "f1" in res
    assert "review_workload" in res
    assert "total_cost" in res
    assert "optimal_points" in res

    opt = res["optimal_points"]
    assert "f1_optimal" in opt
    assert "cost_optimal" in opt
    assert "default_0_5" in opt

    # As threshold increases, review workload should decrease or stay equal
    workloads = res["review_workload"]
    assert workloads[0] >= workloads[-1]
