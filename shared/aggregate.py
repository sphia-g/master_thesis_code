from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class AttentionMIL(nn.Module):
    """Attention-based MIL: aggregates instance-level probabilities to patient level."""

    def __init__(self, hidden_dim: int = 16, epochs: int = 50, lr: float = 0.01):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1), nn.Softmax(dim=0)
        )
        self.classifier = nn.Linear(1, 1)
        self.epochs = epochs
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.attention(x)
        return torch.sigmoid(self.classifier(torch.mm(A.t(), x)))

    def fit(self, probs: np.ndarray, labels: np.ndarray, pids: np.ndarray) -> "AttentionMIL":
        df = pd.DataFrame({"prob": probs, "label": labels, "pid": pids})
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        self.train()
        for _ in range(self.epochs):
            for _, group in df.groupby("pid"):
                x = torch.tensor(group["prob"].values, dtype=torch.float32).view(-1, 1)
                y = torch.tensor([[group["label"].iloc[0]]], dtype=torch.float32)
                optimizer.zero_grad()
                criterion(self(x), y).backward()
                optimizer.step()
        return self

    def predict_proba(self, probs: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (patient_probs, unique_sorted_pids)."""
        df = pd.DataFrame({"prob": probs, "pid": pids})
        self.eval()
        result = {}
        with torch.no_grad():
            for pid, group in df.groupby("pid"):
                x = torch.tensor(group["prob"].values, dtype=torch.float32).view(-1, 1)
                result[pid] = self(x).item()
        unique_pids = np.array(sorted(result))
        return np.array([result[p] for p in unique_pids]), unique_pids


def _majority(probs: np.ndarray, labels: np.ndarray, pids: np.ndarray) -> tuple:
    patient = (
        pd.DataFrame({"prob": probs, "label": labels, "pid": pids})
        .groupby("pid")
        .agg({"prob": "mean", "label": "first"})
    )
    p = patient["prob"].values.astype(float)
    return (p >= 0.5).astype(int), patient["label"].values.astype(int), p


def aggregate_patients(
    method: str,
    train_prob: np.ndarray,
    train_true: np.ndarray,
    train_pids: np.ndarray,
    test_prob: np.ndarray,
    test_true: np.ndarray,
    test_pids: np.ndarray,
    mil_epochs: int = 50,
    mil_lr: float = 0.01,
) -> tuple[tuple, tuple]:
    """Aggregate instance-level probabilities to patient level.

    Returns ((preds, labels, probs), (preds, labels, probs)) for train and test,
    with patients ordered by sorted patient ID.
    """
    assert method in ("majority", "attention"), f"Unknown aggregation method: {method!r}"

    if method == "majority":
        return _majority(train_prob, train_true, train_pids), _majority(test_prob, test_true, test_pids)

    mil = AttentionMIL(epochs=mil_epochs, lr=mil_lr).fit(train_prob, train_true, train_pids)

    def _predict(probs, labels, pids):
        patient_probs, unique_pids = mil.predict_proba(probs, pids)
        patient_labels = (
            pd.DataFrame({"label": labels, "pid": pids})
            .groupby("pid")["label"]
            .first()
            .loc[unique_pids]
            .values.astype(int)
        )
        return (patient_probs >= 0.5).astype(int), patient_labels, patient_probs

    return _predict(train_prob, train_true, train_pids), _predict(test_prob, test_true, test_pids)
