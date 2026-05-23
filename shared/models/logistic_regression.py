import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression as _SklearnLR


class LogisticRegression(nn.Module):
    """Thin PyTorch wrapper around sklearn's LogisticRegression.

    Fitted via sklearn's L-BFGS solver. ``forward()`` calls ``predict_proba``
    and returns class-1 probabilities as a tensor.

    Parameters
    ----------
    weight_decay : float
        L2 regularisation strength; mapped to ``C = 1 / weight_decay``.
    max_iter : int
        Maximum solver iterations (default: 1000).
    """

    def __init__(self, weight_decay: float = 5.0, max_iter: int = 1000):
        super().__init__()
        C = 1.0 / weight_decay if weight_decay > 0.0 else 1e9
        self._model = _SklearnLR(C=C, max_iter=max_iter)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)
