import torch
from torch import nn
from sklearn.dummy import DummyClassifier


class MajorityClassifier(nn.Module):
    """Thin PyTorch wrapper around sklearn's DummyClassifier(strategy='prior').

    Always predicts the class prior probability. Simple baseline to verify whether sophisticated models add value.
    """

    def __init__(self):
        super().__init__()
        self._model = DummyClassifier(strategy="prior")

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)
