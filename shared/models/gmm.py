import numpy as np
import torch
from torch import nn
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel(nn.Module):
    """Thin PyTorch wrapper around two sklearn GaussianMixture models.

    Fits one GMM per class and classifies via Bayes' rule:
    P(y=1|x) ∝ P(x|y=1) * P(y=1)

    Parameters
    ----------
    n_components_per_class : int
        Number of Gaussian components per class (default: 3).
    covariance_type : str
        'full', 'tied', 'diag', or 'spherical' (default: 'diag').
    max_iter : int
        Maximum EM iterations (default: 100).
    tol : float
        Convergence tolerance (default: 1e-3).
    reg_covar : float
        Regularisation added to covariance diagonal (default: 1e-6).
    """

    def __init__(self, n_components_per_class: int = 3, covariance_type: str = 'diag',
                 max_iter: int = 100, tol: float = 1e-3, reg_covar: float = 1e-6):
        super().__init__()
        self.n_components_per_class = int(n_components_per_class)
        self.covariance_type = covariance_type
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self._gmm0 = None
        self._gmm1 = None
        self._log_prior0 = None
        self._log_prior1 = None

    def _fit_single_class(self, X_class: np.ndarray) -> GaussianMixture:
        # Try safer settings first for numerically fragile folds.
        n_samples = int(X_class.shape[0])
        assert n_samples > 0, "Cannot fit GMM for an empty class."

        max_components = min(self.n_components_per_class, n_samples)
        reg_candidates = [
            self.reg_covar,
            self.reg_covar * 10.0,
            self.reg_covar * 100.0,
            self.reg_covar * 1000.0,
            1e-2,
        ]

        last_error = None
        for n_components in range(max_components, 0, -1):
            for reg_covar in reg_candidates:
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    reg_covar=reg_covar,
                )
                try:
                    model.fit(X_class)
                    return model
                except ValueError as exc:
                    last_error = exc

        raise ValueError(
            "GMM fitting failed for all fallback settings. "
            f"n_samples={n_samples}, requested_components={self.n_components_per_class}, "
            f"base_reg_covar={self.reg_covar}."
        ) from last_error

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X_np = X.cpu().numpy().astype(np.float64, copy=False)
        y_np = y.cpu().numpy().ravel()
        X0 = X_np[y_np == 0]
        X1 = X_np[y_np == 1]
        assert len(X0) > 0 and len(X1) > 0, "Both classes are required to fit class-conditional GMMs."

        self._gmm0 = self._fit_single_class(X0)
        self._gmm1 = self._fit_single_class(X1)
        self._log_prior0 = np.log(np.mean(y_np == 0))
        self._log_prior1 = np.log(np.mean(y_np == 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_np = x.detach().cpu().numpy()
        log_p0 = self._gmm0.score_samples(X_np) + self._log_prior0  # log P(x|0)*P(0)
        log_p1 = self._gmm1.score_samples(X_np) + self._log_prior1  # log P(x|1)*P(1)
        prob1 = 1.0 / (1.0 + np.exp(log_p0 - log_p1))
        return torch.tensor(prob1, dtype=torch.float32, device=x.device).unsqueeze(1)
