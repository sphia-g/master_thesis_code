import numpy as np
from tabpfn import TabPFNClassifier


class TabPFN:
    """Thin wrapper around TabPFNClassifier with batched predict_proba.

    TabPFN is a sklearn-compatible pre-trained in-context learner.  Inference
    on large datasets requires batching because the attention mechanism scales
    quadratically with the number of test samples.

    Parameters
    ----------
    batch_size : int
        Number of test samples per predict_proba call (default: 5000).
    ignore_pretraining_limits : bool
        Pass True to allow datasets larger than TabPFN's training regime.
    """

    def __init__(self, batch_size: int = 5000, ignore_pretraining_limits: bool = True):
        assert batch_size > 0, f"batch_size must be > 0, got {batch_size}"
        self._model = TabPFNClassifier(ignore_pretraining_limits=ignore_pretraining_limits)
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        chunks = [
            self._model.predict_proba(X[i : i + self.batch_size])
            for i in range(0, len(X), self.batch_size)
        ]
        probs = np.concatenate(chunks, axis=0)
        assert probs.ndim == 2 and probs.shape[1] == 2, (
            f"Expected predict_proba shape (n_samples, 2), got {probs.shape}"
        )
        return probs
