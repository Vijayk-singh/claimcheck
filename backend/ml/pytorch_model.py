"""
ClaimRisk AI - PyTorch Baseline Neural Network Classifier.

Provides a deep neural network architecture and scikit-learn compatible classifier
wrapper for healthcare claim risk prediction, including:
- Multi-layer perceptron with Batch Normalization, Dropout, and ReLU activations.
- Class-weighted Binary Cross-Entropy Loss to handle class imbalance.
- Early stopping based on validation PR-AUC / loss.
- Serialization and deserialization (.pt / .pth / joblib).
- Scikit-learn estimator interface (fit, predict, predict_proba).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("claimrisk.pytorch_model")


class ClaimRiskNN(nn.Module):
    """Deep neural network architecture for healthcare claim review classification.
    
    Architecture:
        Input(D) -> Linear(D, 64) -> BatchNorm1d -> ReLU -> Dropout(0.2)
                 -> Linear(64, 32) -> BatchNorm1d -> ReLU -> Dropout(0.2)
                 -> Linear(32, 16) -> ReLU
                 -> Linear(16, 1) (logits)
    """

    def __init__(self, input_dim: int = 28, dropout_rate: float = 0.20) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (shape: [batch_size, 1])."""
        return self.network(x)


class PyTorchClaimRiskClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible classifier wrapper around ClaimRiskNN.
    
    Allows seamless integration with scikit-learn evaluation metrics,
    cross-validation, and pipeline architectures.
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout_rate: float = 0.20,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 18,
        patience: int = 4,
        pos_weight: Optional[float] = None,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.pos_weight = pos_weight
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_: Optional[ClaimRiskNN] = None
        self.classes_ = np.array([0, 1])
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    def fit(
        self,
        X: Union[np.ndarray, Any],
        y: Union[np.ndarray, Any],
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
    ) -> PyTorchClaimRiskClassifier:
        """Fit the PyTorch neural network on training data."""
        self._set_seed()

        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.input_dim = X.shape[1]
        self.model_ = ClaimRiskNN(input_dim=self.input_dim, dropout_rate=self.dropout_rate)
        self.model_.to(self.device)

        # Class imbalance weighting
        if self.pos_weight is None:
            n_neg = float(np.sum(y == 0))
            n_pos = float(np.sum(y == 1))
            calculated_pos_weight = (n_neg / max(n_pos, 1.0)) if n_pos > 0 else 1.0
        else:
            calculated_pos_weight = float(self.pos_weight)

        pos_weight_tensor = torch.tensor([calculated_pos_weight], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
        )

        # Prepare PyTorch datasets
        train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            if hasattr(X_val, "values"):
                X_val = X_val.values
            if hasattr(y_val, "values"):
                y_val = y_val.values
            X_val_t = torch.tensor(np.asarray(X_val, dtype=np.float32), device=self.device)
            y_val_t = torch.tensor(
                np.asarray(y_val, dtype=np.float32).reshape(-1, 1), device=self.device
            )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model_.train()
        for epoch in range(self.epochs):
            total_train_loss = 0.0
            num_batches = 0

            self.model_.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / max(num_batches, 1)
            self.history_["train_loss"].append(avg_train_loss)

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                self.history_["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            else:
                scheduler.step(avg_train_loss)

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.model_.eval()
        return self

    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Predict class probabilities for X (returns shape [N, 2])."""
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        if hasattr(X, "values"):
            X = X.values
        X_arr = np.asarray(X, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_arr).to(self.device)
            logits = self.model_(tensor_x).cpu().numpy().reshape(-1)
            prob_1 = 1.0 / (1.0 + np.exp(-logits))
            prob_1 = np.clip(prob_1, 1e-7, 1.0 - 1e-7)
            prob_0 = 1.0 - prob_1

        return np.column_stack([prob_0, prob_1])

    def predict(self, X: Union[np.ndarray, Any], threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels using the specified decision threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(int)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save PyTorch model weights and configuration to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model_.state_dict() if self.model_ is not None else None,
            "input_dim": self.input_dim,
            "dropout_rate": self.dropout_rate,
            "pos_weight": self.pos_weight,
            "history": self.history_,
        }
        torch.save(checkpoint, str(path))
        logger.info(f"Saved PyTorch model checkpoint to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> PyTorchClaimRiskClassifier:
        """Load saved PyTorch model checkpoint from disk."""
        path = Path(file_path)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(path), map_location=dev, weights_only=False)

        classifier = cls(
            input_dim=checkpoint["input_dim"],
            dropout_rate=checkpoint.get("dropout_rate", 0.20),
            pos_weight=checkpoint.get("pos_weight", None),
            device=dev,
        )
        classifier.model_ = ClaimRiskNN(input_dim=classifier.input_dim, dropout_rate=classifier.dropout_rate)
        if checkpoint["state_dict"] is not None:
            classifier.model_.load_state_dict(checkpoint["state_dict"])
        classifier.model_.to(dev)
        classifier.model_.eval()
        classifier.history_ = checkpoint.get("history", {})
        return classifier
