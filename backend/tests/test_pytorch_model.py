"""
Tests for PyTorch Baseline Classifier.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ml.pytorch_model import ClaimRiskNN, PyTorchClaimRiskClassifier


def test_claim_risk_nn_architecture() -> None:
    """Verify neural network forward pass logits and dimensionality."""
    model = ClaimRiskNN(input_dim=28, dropout_rate=0.20)
    x = torch.randn(8, 28)
    logits = model(x)
    assert logits.shape == (8, 1)


def test_pytorch_classifier_fit_predict() -> None:
    """Test classifier fitting, probability calibration, and prediction."""
    np.random.seed(42)
    X = np.random.randn(100, 28).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = PyTorchClaimRiskClassifier(
        epochs=5,
        batch_size=32,
        learning_rate=0.01,
        random_state=42,
    )
    clf.fit(X, y)

    # Test predict_proba
    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    np.testing.assert_allclose(probs[:, 0] + probs[:, 1], 1.0, atol=1e-5)

    # Test predict
    preds = clf.predict(X, threshold=0.50)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_pytorch_classifier_save_load() -> None:
    """Test serialization and checkpoint reloading."""
    np.random.seed(42)
    X = np.random.randn(60, 28).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    clf = PyTorchClaimRiskClassifier(epochs=3, batch_size=32, random_state=42)
    clf.fit(X, y)
    orig_probs = clf.predict_proba(X[:10])

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_pt.pt"
        clf.save(model_path)
        assert model_path.exists()

        loaded_clf = PyTorchClaimRiskClassifier.load(model_path, device="cpu")
        loaded_probs = loaded_clf.predict_proba(X[:10])
        np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-5)
