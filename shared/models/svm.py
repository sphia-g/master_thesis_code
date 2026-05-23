import torch
from torch import nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


class SupportVectorMachine(nn.Module):
    """PyTorch wrapper around sklearn SVM estimators.

    Parameters
    ----------
    C : float
        Regularisation parameter (default: 1.0).
    kernel : str
        Kernel type: 'linear', 'rbf', 'poly', 'sigmoid' (default: 'linear').
    gamma : float or str
        Kernel coefficient for 'rbf', 'poly', 'sigmoid' (default: 'auto').
    """

    def __init__(self, C: float = 1.0, kernel: str = 'linear', gamma: str = 'auto'):
        super().__init__()
        if kernel == 'linear':
            base = LinearSVC(C=C, dual=False, max_iter=10000)
            self._model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        else:
            self._model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(X.cpu().numpy(), y.cpu().numpy().ravel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self._model.predict_proba(x.detach().cpu().numpy())[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device).unsqueeze(1)
