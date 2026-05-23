import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier


class RandomForest(nn.Module):
    """Thin PyTorch wrapper around sklearn's RandomForestClassifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default: 10).
    depth : int
        Maximum depth of each tree (default: 3).
    max_features : int or str
        Features considered per split: 'sqrt', 'log2', int, or None (default: 'sqrt').
    bootstrap : bool
        Whether to use bootstrap samples (default: True).
    """

    def __init__(self, n_estimators: int = 10, depth: int = 3,
                 max_features: str = 'sqrt', bootstrap: bool = True):
        super().__init__()
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=depth,
            max_features=max_features,
            bootstrap=bootstrap,
        )

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)

    @property
    def feature_importances(self) -> torch.Tensor:
        """Gini-based feature importances from sklearn, shape (n_features,)."""
        return torch.tensor(self._model.feature_importances_, dtype=torch.float32)

    