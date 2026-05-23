import torch
from torch import nn
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors(nn.Module):
    """Thin PyTorch wrapper around sklearn's KNeighborsClassifier.

    Parameters
    ----------
    k : int
        Number of neighbours (default: 5).
    """

    def __init__(self, k: int = 5):
        super().__init__()
        self._model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)

