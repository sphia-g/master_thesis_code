import torch
from torch import nn
from typing import Tuple
from sklearn.neural_network import MLPClassifier


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        alpha: float = 0.0001,
    ):
        super().__init__()
        self._model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)