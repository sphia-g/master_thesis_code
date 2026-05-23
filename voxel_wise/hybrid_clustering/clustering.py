"""
Clustering utilities for hybrid cluster-filtered classification.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)


def find_optimal_clusters(
    X: np.ndarray,
    k_range: range = range(2, 8),
    method: str = "kmeans",
    random_state: int = 42
) -> Tuple[dict, dict]:
    """Find optimal number of clusters using AIC, BIC, and silhouette scores.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (should be scaled)
    k_range : range
        Range of k values to test
    method : str
        'kmeans' or 'gmm'
    random_state : int
        Random seed
    
    Returns
    -------
    metrics : dict
        {k: {'aic': float, 'bic': float, 'silhouette': float, 'dbi': float}}
    best_k : dict
        {'aic': int, 'bic': int, 'silhouette': int, 'dbi': int}
    """
    metrics = {}
    
    logger.info(f"Testing k in {list(k_range)} using {method}")
    
    for k in k_range:
        if method == "kmeans":
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(X)
            
            # Compute BIC/AIC approximation for K-means (using inertia)
            inertia = model.inertia_
            n_samples, n_features = X.shape
            n_params = k * n_features + k  # centroids + cluster assignments
            
            bic = inertia + n_params * np.log(n_samples)
            aic = inertia + 2 * n_params
            
        elif method == "gmm":
            model = GaussianMixture(n_components=k, random_state=random_state, n_init=10)
            model.fit(X)
            labels = model.predict(X)
            
            bic = model.bic(X)
            aic = model.aic(X)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Silhouette score (higher is better, range [-1, 1])
        sil_score = silhouette_score(X, labels, sample_size=min(10000, len(X)))
        
        # Davies-Bouldin Index (lower is better)
        dbi = davies_bouldin_score(X, labels)
        
        metrics[k] = {
            'aic': float(aic),
            'bic': float(bic),
            'silhouette': float(sil_score),
            'dbi': float(dbi)
        }
        
        logger.info(f"  k={k}: AIC={aic:.2f}, BIC={bic:.2f}, Silhouette={sil_score:.3f}, DBI={dbi:.3f}")
    
    # Find best k for each metric
    best_k = {
        'aic': min(metrics.keys(), key=lambda k: metrics[k]['aic']),
        'bic': min(metrics.keys(), key=lambda k: metrics[k]['bic']),
        'silhouette': max(metrics.keys(), key=lambda k: metrics[k]['silhouette']),
        'dbi': min(metrics.keys(), key=lambda k: metrics[k]['dbi'])
    }
    
    logger.info(f"\nOptimal k: AIC={best_k['aic']}, BIC={best_k['bic']}, "
                f"Silhouette={best_k['silhouette']}, DBI={best_k['dbi']}")
    
    return metrics, best_k


def fit_clusterer(
    X: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    random_state: int = 42
):
    """Fit clustering model.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (should be scaled)
    n_clusters : int
        Number of clusters
    method : str
        'kmeans' or 'gmm'
    random_state : int
        Random seed
    
    Returns
    -------
    clusterer : fitted model
        KMeans or GaussianMixture instance
    """
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif method == "gmm":
        clusterer = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=10)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Fitting {method} with {n_clusters} clusters...")
    clusterer.fit(X)
    
    return clusterer


def get_discriminative_cluster(
    cluster_labels: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    criterion: str = "percentage_diff"
) -> Tuple[int, dict]:
    """Identify cluster with largest LGG/HGG separation.
    
    Parameters
    ----------
    cluster_labels : np.ndarray, shape (n_voxels,)
        Cluster assignment for each voxel
    y : np.ndarray, shape (n_voxels,)
        Class labels (0=LGG, 1=HGG)
    patient_ids : np.ndarray, shape (n_voxels,)
        Patient ID for each voxel
    criterion : str
        'percentage_diff': absolute difference in percentage of LGG vs HGG voxels
        'chi2': chi-square statistic
    
    Returns
    -------
    best_cluster : int
        Cluster index with largest separation
    cluster_stats : dict
        Statistics for each cluster
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_stats = {}
    
    logger.info(f"\nAnalyzing cluster separation (criterion: {criterion}):")
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_y = y[cluster_mask]
        
        n_total = len(cluster_y)
        n_lgg = np.sum(cluster_y == 0)
        n_hgg = np.sum(cluster_y == 1)
        
        pct_lgg = n_lgg / n_total * 100 if n_total > 0 else 0
        pct_hgg = n_hgg / n_total * 100 if n_total > 0 else 0
        
        # Per-patient statistics
        cluster_patients = patient_ids[cluster_mask]
        unique_patients = np.unique(cluster_patients)
        n_patients = len(unique_patients)
        
        # Convert cluster_id to int for JSON serialization
        cluster_stats[int(cluster_id)] = {
            'n_voxels': int(n_total),
            'n_lgg_voxels': int(n_lgg),
            'n_hgg_voxels': int(n_hgg),
            'pct_lgg': float(pct_lgg),
            'pct_hgg': float(pct_hgg),
            'percentage_diff': float(abs(pct_lgg - pct_hgg)),
            'n_patients': int(n_patients)
        }
        
        logger.info(f"  Cluster {cluster_id}: {n_total:,} voxels from {n_patients} patients | "
                    f"LGG: {pct_lgg:.1f}%, HGG: {pct_hgg:.1f}% | "
                    f"Diff: {abs(pct_lgg - pct_hgg):.1f}%")
    
    # Select best cluster
    if criterion == "percentage_diff":
        best_cluster = max(cluster_stats.keys(), key=lambda c: cluster_stats[c]['percentage_diff'])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    return int(best_cluster), cluster_stats