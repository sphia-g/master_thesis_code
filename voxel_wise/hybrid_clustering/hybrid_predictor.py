"""
Hybrid cluster-filtered predictor with default fallback.
"""

from __future__ import annotations

import logging
import numpy as np
import torch.nn as nn

from ...shared.train import train
from ...shared.evaluate import get_predictions_batch
from .clustering import fit_clusterer, get_discriminative_cluster

logger = logging.getLogger(__name__)


class HybridClusterPredictor:
    """Cluster-filtered classifier: trains on the most discriminative voxel cluster
    and predicts the minority class for voxels outside it."""

    def __init__(
        self,
        base_classifier,
        device,
        n_clusters: int = 3,
        method: str = "kmeans",
        criterion: str = "percentage_diff",
        min_cluster_voxels: int = 10,
        random_state: int = 42,
    ):
        self.base_classifier = base_classifier
        self.device = device
        self.n_clusters = n_clusters
        self.method = method
        self.criterion = criterion
        self.min_cluster_voxels = min_cluster_voxels
        self.random_state = random_state

        self.clusterer = None
        self.selected_cluster = None
        self.default_prediction = None
        self.cluster_stats = None

    def fit(self, X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray) -> "HybridClusterPredictor":
        """Fit clustering + classifier on already-scaled training data."""
        self.clusterer = fit_clusterer(X, n_clusters=self.n_clusters, method=self.method, random_state=self.random_state)
        cluster_labels = self.clusterer.predict(X)

        self.selected_cluster, self.cluster_stats = get_discriminative_cluster(
            cluster_labels=cluster_labels, y=y, patient_ids=patient_ids, criterion=self.criterion
        )

        cluster_mask = cluster_labels == self.selected_cluster
        cluster_y = y[cluster_mask]
        self.default_prediction = int(np.argmin([np.sum(cluster_y == 0), np.sum(cluster_y == 1)]))

        logger.info(f"Selected cluster {self.selected_cluster} | default={self.default_prediction} | "
                    f"{cluster_mask.sum():,}/{len(X):,} voxels in cluster")

        X_cl, y_cl = X[cluster_mask], y[cluster_mask]
        if isinstance(self.base_classifier, nn.Module):
            train(self.base_classifier, X_cl, y_cl, self.device)
        else:
            self.base_classifier.fit(X_cl, y_cl)

        return self

    def _run(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (preds, probs) for all voxels."""
        in_cluster = self.clusterer.predict(X) == self.selected_cluster
        n_in = in_cluster.sum()

        preds = np.full(len(X), self.default_prediction, dtype=int)
        probs = np.full(len(X), float(self.default_prediction))

        if n_in >= self.min_cluster_voxels:
            X_cl = X[in_cluster]
            if isinstance(self.base_classifier, nn.Module):
                preds[in_cluster], probs[in_cluster] = get_predictions_batch(
                    self.base_classifier, X_cl, self.device, batch_size=10000
                )
            else:
                p = self.base_classifier.predict_proba(X_cl)[:, 1]
                preds[in_cluster] = (p >= 0.5).astype(int)
                probs[in_cluster] = p
        else:
            logger.warning(f"Only {n_in} cluster voxels (< {self.min_cluster_voxels}); using default for all.")

        return preds, probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._run(X)[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._run(X)[1]

    def get_cluster_info(self) -> dict:
        return {
            'n_clusters': self.n_clusters,
            'method': self.method,
            'selected_cluster': int(self.selected_cluster),
            'default_prediction': int(self.default_prediction),
            'cluster_stats': self.cluster_stats,
        }