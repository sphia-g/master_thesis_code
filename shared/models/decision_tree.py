import torch
from torch import nn
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(nn.Module):
    """Thin PyTorch wrapper around sklearn's DecisionTreeClassifier.

    Parameters
    ----------
    depth : int
        Maximum tree depth (default: 3).
    """

    def __init__(self, depth: int = 3):
        super().__init__()
        self._model = DecisionTreeClassifier(max_depth=depth)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)
