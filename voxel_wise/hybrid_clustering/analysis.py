"""
Analysis and visualization for cluster-filtered classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


def plot_cluster_metrics(
    metrics: dict,
    save_path: Path | None = None,
    title_suffix: str = ""
) -> None:
    """Plot AIC, BIC, Silhouette, and DBI scores across different k values.
    
    Parameters
    ----------
    metrics : dict
        {k: {'aic': float, 'bic': float, 'silhouette': float, 'dbi': float}}
    save_path : Optional[Path]
        Path to save figure
    title_suffix : str
        Additional text for title (e.g., "Fold 1")
    """
    k_values = sorted(metrics.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Cluster Selection Metrics{" - " + title_suffix if title_suffix else ""}', 
                 fontsize=14, fontweight='bold')
    
    # Color scheme from your plots
    color_purple = '#4c3fbd'
    color_lavender = '#b39ddb'
    color_light_purple = '#edc2f1'
    
    # AIC
    aic_values = [metrics[k]['aic'] for k in k_values]
    axes[0, 0].plot(k_values, aic_values, 'o-', linewidth=2, markersize=8, color=color_purple)
    axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0, 0].set_ylabel('AIC', fontsize=11)
    axes[0, 0].set_title('Akaike Information Criterion (lower is better)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # BIC
    bic_values = [metrics[k]['bic'] for k in k_values]
    axes[0, 1].plot(k_values, bic_values, 'o-', linewidth=2, markersize=8, color=color_lavender)
    axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0, 1].set_ylabel('BIC', fontsize=11)
    axes[0, 1].set_title('Bayesian Information Criterion (lower is better)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Silhouette
    sil_values = [metrics[k]['silhouette'] for k in k_values]
    axes[1, 0].plot(k_values, sil_values, 'o-', linewidth=2, markersize=8, color=color_light_purple)
    axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1, 0].set_ylabel('Silhouette Score', fontsize=11)
    axes[1, 0].set_title('Silhouette Score (higher is better)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Davies-Bouldin Index
    dbi_values = [metrics[k]['dbi'] for k in k_values]
    axes[1, 1].plot(k_values, dbi_values, 'o-', linewidth=2, markersize=8, color='#6a5acd')
    axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1, 1].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1, 1].set_title('Davies-Bouldin Index (lower is better)', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cluster metrics plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_cluster_separation(
    cluster_stats: dict,
    selected_cluster: int,
    save_path: Path | None = None,
    title_suffix: str = ""
) -> None:
    """Plot bar chart showing LGG/HGG distribution across clusters.
    
    Parameters
    ----------
    cluster_stats : dict
        Statistics per cluster from get_discriminative_cluster()
    selected_cluster : int
        The selected discriminative cluster
    save_path : Optional[Path]
        Path to save figure
    title_suffix : str
        Additional text for title
    """
    cluster_ids = sorted(cluster_stats.keys())
    n_clusters = len(cluster_ids)
    
    # Prepare data
    lgg_pcts = [cluster_stats[c]['pct_lgg'] for c in cluster_ids]
    hgg_pcts = [cluster_stats[c]['pct_hgg'] for c in cluster_ids]
    n_voxels = [cluster_stats[c]['n_voxels'] for c in cluster_ids]
    
    # Calculate total voxels for percentage calculation
    total_voxels = sum(n_voxels)
    pct_of_total = [(n / total_voxels * 100) for n in n_voxels]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f'Cluster-Class Distribution{" - " + title_suffix if title_suffix else ""}',
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Stacked percentage bar plot
    # Same colors for all, different alpha for selected vs non-selected
    x = np.arange(n_clusters)
    width = 0.6
    
    # Use same colors with different alpha
    for i, c in enumerate(cluster_ids):
        alpha = 1.0 if c == selected_cluster else 0.3
        
        # LGG bar
        ax1.bar(i, lgg_pcts[i], width, color='#b39ddb', alpha=alpha,
                edgecolor='black' if c == selected_cluster else 'none', 
                linewidth=2 if c == selected_cluster else 0)
        
        # HGG bar (stacked on top)
        ax1.bar(i, hgg_pcts[i], width, bottom=lgg_pcts[i], color='#4c3fbd', alpha=alpha,
                edgecolor='black' if c == selected_cluster else 'none', 
                linewidth=2 if c == selected_cluster else 0)
    
    # Add legend manually
    legend_elements = [
        Patch(facecolor='#b39ddb', edgecolor='black', label='LGG'),
        Patch(facecolor='#4c3fbd', edgecolor='black', label='HGG')
    ]
    ax1.legend(handles=legend_elements)
    
    ax1.set_ylabel('Percentage (%)', fontsize=11)
    ax1.set_xlabel('Cluster', fontsize=11)
    ax1.set_title('LGG/HGG Distribution per Cluster (100% stacked)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{c}' for c in cluster_ids])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add percentage labels
    for i, (c, lgg, hgg) in enumerate(zip(cluster_ids, lgg_pcts, hgg_pcts)):
        if lgg > 5:
            ax1.text(i, lgg/2, f'{lgg:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        if hgg > 5:
            ax1.text(i, lgg + hgg/2, f'{hgg:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Voxel counts - grey except selected in your color
    colors_bar = ['#edc2f1' if c == selected_cluster else 'darkgrey' for c in cluster_ids]
    ax2.bar(x, n_voxels, width, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Voxels', fontsize=11)
    ax2.set_xlabel('Cluster', fontsize=11)
    ax2.set_title('Voxel Count per Cluster', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{c}' for c in cluster_ids])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add count labels WITH PERCENTAGE
    for i, (count, pct) in enumerate(zip(n_voxels, pct_of_total)):
        ax2.text(i, count, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Add text annotation for selected cluster - always same background color
    separation_diff = cluster_stats[selected_cluster]['percentage_diff']
    n_patients_in_cluster = cluster_stats[selected_cluster]['n_patients']
    
    ax1.text(0.02, 0.98, 
             f'Selected Cluster: C{selected_cluster}\n'
             f'Separation: {separation_diff:.1f}%\n'
             f'Patients: {n_patients_in_cluster}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cluster separation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fold_summary(
    fold_results: list,
    save_path: Path | None = None
) -> None:
    """Plot summary of selected clusters and performance across folds.
    
    Parameters
    ----------
    fold_results : list
        List of dicts with keys: 'fold', 'selected_cluster', 'test_accuracy', etc.
    save_path : Optional[Path]
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Validation Summary: Cluster Selection & Performance', 
                 fontsize=14, fontweight='bold')
    
    folds = [r['fold'] for r in fold_results]
    selected_clusters = [r['selected_cluster'] for r in fold_results]
    test_accs = [r['test_accuracy'] for r in fold_results]
    
    # Plot 1: Selected cluster per fold
    ax1 = axes[0]
    colors = plt.cm.tab10(np.array(selected_clusters) / max(selected_clusters))
    ax1.bar(folds, selected_clusters, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_ylabel('Selected Cluster ID', fontsize=11)
    ax1.set_title('Discriminative Cluster Selected per Fold', fontsize=11)
    ax1.set_xticks(folds)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add cluster ID labels
    for fold, cluster in zip(folds, selected_clusters):
        ax1.text(fold, cluster, f'C{cluster}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Test accuracy per fold
    ax2 = axes[1]
    ax2.plot(folds, test_accs, 'o-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('Test Accuracy', fontsize=11)
    ax2.set_title('Test Accuracy per Fold', fontsize=11)
    ax2.set_xticks(folds)
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    mean_acc = np.mean(test_accs)
    ax2.axhline(mean_acc, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Mean: {mean_acc:.3f}')
    ax2.legend()
    
    # Add accuracy labels
    for fold, acc in zip(folds, test_accs):
        ax2.text(fold, acc, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved fold summary plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def plot_patient_cluster_heatmap(
    cluster_labels: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    n_clusters: int,
    save_path: Path | None = None,
    title_suffix: str = ""
) -> None:
    """Plot heatmap showing percentage of each patient's tumor in each cluster.
    
    Patients are sorted by their actual class labels (0=LGG, 1=HGG from dataset).
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster assignment for each voxel
    y : np.ndarray
        Class labels (0=LGG, 1=HGG) - from dataset, not clustering
    patient_ids : np.ndarray
        Patient ID for each voxel
    n_clusters : int
        Number of clusters
    save_path : Optional[Path]
        Path to save figure
    title_suffix : str
        Additional text for title
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    # Get patient labels (0=LGG, 1=HGG from original dataset labels, NOT from clustering)
    patient_labels = np.array([y[patient_ids == p][0] for p in unique_patients])
    
    # Calculate percentage matrix: patients x clusters
    pct_matrix = np.zeros((n_patients, n_clusters))
    
    for i, patient_id in enumerate(unique_patients):
        patient_mask = patient_ids == patient_id
        patient_clusters = cluster_labels[patient_mask]
        patient_n_voxels = len(patient_clusters)
        
        for c in range(n_clusters):
            n_in_cluster = np.sum(patient_clusters == c)
            pct_matrix[i, c] = (n_in_cluster / patient_n_voxels * 100)
    
    # Sort patients by class (LGG=0 first, then HGG=1)
    sort_idx = np.argsort(patient_labels)
    pct_matrix_sorted = pct_matrix[sort_idx]
    patient_labels_sorted = patient_labels[sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Heatmap
    im = ax.imshow(pct_matrix_sorted, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Patient ID (sorted by grade)', fontsize=12, fontweight='bold')
    ax.set_title(f'Patient Tumor Composition by Cluster (normalized){" - " + title_suffix if title_suffix else ""}',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f'C{c}' for c in range(n_clusters)])
    
    # Patient IDs start from 1
    y_tick_step = max(1, n_patients // 20)
    y_ticks = range(0, n_patients, y_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([i + 1 for i in y_ticks])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% of Patient Voxels', fontsize=11)
    
    # Add class divider line and labels
    n_lgg = np.sum(patient_labels_sorted == 0)
    if n_lgg > 0 and n_lgg < n_patients:
        ax.axhline(n_lgg - 0.5, color='#4c3fbd', linewidth=3, linestyle='--')
        # Place labels with same style as bar plots
        ax.text(-0.8, n_lgg / 2, 'LGG', rotation=90, va='center', ha='center', 
                fontsize=11, fontweight='bold', color='#4c3fbd',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.text(-0.8, n_lgg + (n_patients - n_lgg) / 2, 'HGG', rotation=90, va='center', ha='center',
                fontsize=11, fontweight='bold', color='#4c3fbd',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))  
          
    # Add explanation text
    ax.text(0.5, -0.08, 
            'Note: Patients sorted by original dataset labels (LGG=0, HGG=1), not by clustering results',
            transform=ax.transAxes, fontsize=9, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved patient cluster heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_patient_cluster_bars(
    cluster_labels: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    n_clusters: int,
    save_path: Path | None = None,
    title_suffix: str = ""
) -> None:
    """Plot stacked bar charts showing normalized and absolute cluster distributions per patient.
    
    Patients are sorted by their actual class labels (0=LGG, 1=HGG from dataset).
    
    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster assignment for each voxel
    y : np.ndarray
        Class labels (0=LGG, 1=HGG) - from dataset, not clustering
    patient_ids : np.ndarray
        Patient ID for each voxel
    n_clusters : int
        Number of clusters
    save_path : Optional[Path]
        Path to save figure
    title_suffix : str
        Additional text for title
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    # Get patient labels (from original dataset)
    patient_labels = np.array([y[patient_ids == p][0] for p in unique_patients])
    
    # Calculate percentage matrix and absolute count matrix: patients x clusters
    pct_matrix = np.zeros((n_patients, n_clusters))
    count_matrix = np.zeros((n_patients, n_clusters))
    
    for i, patient_id in enumerate(unique_patients):
        patient_mask = patient_ids == patient_id
        patient_clusters = cluster_labels[patient_mask]
        patient_n_voxels = len(patient_clusters)
        
        for c in range(n_clusters):
            n_in_cluster = np.sum(patient_clusters == c)
            count_matrix[i, c] = n_in_cluster
            pct_matrix[i, c] = (n_in_cluster / patient_n_voxels * 100)
    
    # Sort patients by class (LGG first, then HGG) based on ORIGINAL labels
    sort_idx = np.argsort(patient_labels)
    pct_matrix_sorted = pct_matrix[sort_idx]
    count_matrix_sorted = count_matrix[sort_idx]
    patient_labels_sorted = patient_labels[sort_idx]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Patient-Level Cluster Distribution{" - " + title_suffix if title_suffix else ""}',
                 fontsize=14, fontweight='bold')
    
    # Patient IDs (starting from 1)
    x = np.arange(n_patients)
    patient_labels_display = np.arange(1, n_patients + 1)
    
    # Diverse color palette with good separation (turquoise, lavender, blue, grey, pink, yellow, purple, orange)
    diverse_colors = [
        "#ab8ae6",  # Purple (C1)
        '#4c3fbd',  # Dark purple (C2)
        '#a6a6a6',  # Grey (C3)
        '#6faed6',  # Blue (C5)
        '#fde68a',  # Yellow (C4)
        "#eec6de",  # Pink (C0)
        '#7fc8c4',  # Turquoise/mint (C6)
        '#b8b3d9',  # Lavender (C7)        
        '#f9a65a',  # Orange (C8)        
        '#66c2a5',  # Teal (C9)
    ]
    
    # Select colors based on number of clusters
    if n_clusters <= len(diverse_colors):
        colors = [diverse_colors[i] for i in range(n_clusters)]
    else:
        colors = diverse_colors[:n_clusters]
    
    # Plot 1: Normalized (% of patient tumor) - STACKED
    bottom_pct = np.zeros(n_patients)
    for c in range(n_clusters):
        ax1.bar(x, pct_matrix_sorted[:, c], bottom=bottom_pct, 
               label=f'C{c}', color=colors[c], alpha=0.9, edgecolor='white', linewidth=0.5)
        bottom_pct += pct_matrix_sorted[:, c]
    
    ax1.set_xlabel('Patient ID (sorted by grade)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('% of Patient Tumor', fontsize=11, fontweight='bold')
    ax1.set_title('Normalized: % of Each Patient\'s Tumor per Cluster (accounts for tumor size)', 
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.5, n_patients - 0.5)
    
    # Legend only in top plot
    ax1.legend(title='Cluster', ncol=n_clusters, loc='upper left', fontsize=10, framealpha=0.95)
    
    # Set x-axis labels
    x_tick_step = max(1, n_patients // 20)
    x_ticks = range(0, n_patients, x_tick_step)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([patient_labels_display[i] for i in x_ticks], fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add class divider
    n_lgg = np.sum(patient_labels_sorted == 0)
    if n_lgg > 0 and n_lgg < n_patients:
        ax1.axvline(n_lgg - 0.5, color='#4c3fbd', linewidth=2.5, linestyle='--', alpha=0.7)
        ax1.text(n_lgg / 2, 95, 'LGG', ha='center', fontsize=11, fontweight='bold', 
                color='#4c3fbd', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax1.text(n_lgg + (n_patients - n_lgg) / 2, 95, 'HGG', ha='center', fontsize=11, 
                fontweight='bold', color='#4c3fbd', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Plot 2: Absolute (voxel counts) - STACKED
    bottom_count = np.zeros(n_patients)
    for c in range(n_clusters):
        ax2.bar(x, count_matrix_sorted[:, c], bottom=bottom_count,
               color=colors[c], alpha=0.9, edgecolor='white', linewidth=0.5)
        bottom_count += count_matrix_sorted[:, c]
    
    ax2.set_xlabel('Patient ID (sorted by grade)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Voxel Count', fontsize=11, fontweight='bold')
    ax2.set_title('Absolute: Total Voxel Counts per Patient (reflects tumor volume)', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlim(-0.5, n_patients - 0.5)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([patient_labels_display[i] for i in x_ticks], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add class divider
    if n_lgg > 0 and n_lgg < n_patients:
        ax2.axvline(n_lgg - 0.5, color='#4c3fbd', linewidth=2.5, linestyle='--', alpha=0.7)
        ax2.text(n_lgg / 2, ax2.get_ylim()[1] * 0.95, 'LGG', ha='center', fontsize=11, 
                fontweight='bold', color='#4c3fbd', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax2.text(n_lgg + (n_patients - n_lgg) / 2, ax2.get_ylim()[1] * 0.95, 'HGG', ha='center', 
                fontsize=11, fontweight='bold', color='#4c3fbd', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved patient cluster bar charts to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
def analyze_cluster_feature_importance(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: list,
    best_cluster: int,
    k: int,
    save_path: Path | None = None,
    title_suffix: str = "",
    top_n: int = 20
) -> dict:
    """Analyze which features are most important for the discriminative cluster.
    
    Uses multiple approaches:
    1. Cluster centroids - features with largest differences from overall mean
    2. ANOVA F-statistic - features that vary most between clusters
    3. Random Forest feature importance - discriminative power
    
    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix
    cluster_labels : np.ndarray
        Cluster assignment for each voxel
    feature_names : list
        Names of features
    best_cluster : int
        The selected discriminative cluster
    k : int
        Number of clusters
    save_path : Optional[Path]
        Path to save figure
    title_suffix : str
        Additional text for title
    top_n : int
        Number of top features to display
    
    Returns
    -------
    feature_scores : Dict
        Dictionary with feature importance metrics
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import f_classif
    
    n_clusters = len(np.unique(cluster_labels))
    n_features = X_scaled.shape[1]
    
    logger.info(f"Analyzing feature importance for cluster {best_cluster} (k={k})...")
    
    # ===================================================================
    # Method 1: Cluster Centroid Differences (for selected cluster)
    # ===================================================================
    overall_mean = np.mean(X_scaled, axis=0)
    cluster_mask = cluster_labels == best_cluster
    cluster_mean = np.mean(X_scaled[cluster_mask], axis=0)
    
    # Distance from overall mean (absolute difference)
    centroid_diff = np.abs(cluster_mean - overall_mean)
    centroid_importance = centroid_diff / np.sum(centroid_diff)  # Normalize
    
    # ===================================================================
    # Method 2: ANOVA F-statistic (all clusters)
    # ===================================================================
    f_scores, p_values = f_classif(X_scaled, cluster_labels)
    anova_importance = f_scores / np.sum(f_scores)  # Normalize
    
    # ===================================================================
    # Method 3: Random Forest Feature Importance
    # ===================================================================
    # Binary classification: selected cluster vs all others
    binary_labels = (cluster_labels == best_cluster).astype(int)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_scaled, binary_labels)
    rf_importance = rf.feature_importances_
    
    # ===================================================================
    # Combine and rank features
    # ===================================================================
    feature_scores = []
    for i, fname in enumerate(feature_names):
        feature_scores.append({
            'feature': fname,
            'centroid_diff': float(centroid_diff[i]),
            'centroid_importance': float(centroid_importance[i]),
            'anova_f': float(f_scores[i]),
            'anova_p': float(p_values[i]),
            'anova_importance': float(anova_importance[i]),
            'rf_importance': float(rf_importance[i]),
            'avg_importance': float((centroid_importance[i] + anova_importance[i] + rf_importance[i]) / 3)
        })
    
    # Sort by average importance
    feature_scores = sorted(feature_scores, key=lambda x: x['avg_importance'], reverse=True)
    
    # ===================================================================
    # Log top features
    # ===================================================================
    logger.info(f"\n   Top {top_n} Most Important Features (Cluster {best_cluster}):")
    logger.info(f"   {'Rank':<6} {'Feature':<25} {'Centroid':<12} {'ANOVA-F':<12} {'RF':<12} {'Avg':<12}")
    logger.info(f"   {'-'*79}")
    
    for rank, fs in enumerate(feature_scores[:top_n], 1):
        logger.info(f"   {rank:<6} {fs['feature']:<25} "
                   f"{fs['centroid_importance']:.6f}   "
                   f"{fs['anova_importance']:.6f}   "
                   f"{fs['rf_importance']:.6f}   "
                   f"{fs['avg_importance']:.6f}")
    
    # ===================================================================
    # Visualize feature importance
    # ===================================================================
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Feature Importance Analysis{" - " + title_suffix if title_suffix else ""}',
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Top features by average importance
        top_features = feature_scores[:top_n]
        feature_labels = [fs['feature'] for fs in top_features]
        avg_scores = [fs['avg_importance'] for fs in top_features]
        
        axes[0, 0].barh(range(top_n), avg_scores, color='#4c3fbd')
        axes[0, 0].set_yticks(range(top_n))
        axes[0, 0].set_yticklabels(feature_labels, fontsize=9)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel('Average Importance Score', fontsize=11)
        axes[0, 0].set_title(f'Top {top_n} Features by Average Importance\n(Cluster {best_cluster}, k={k})', 
                             fontsize=12, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Comparison of methods
        n_compare = min(10, len(top_features))
        x_pos = np.arange(n_compare)
        width = 0.25
        
        centroid_scores = [fs['centroid_importance'] for fs in top_features[:n_compare]]
        anova_scores = [fs['anova_importance'] for fs in top_features[:n_compare]]
        rf_scores = [fs['rf_importance'] for fs in top_features[:n_compare]]
        short_labels = [fs['feature'][:15] for fs in top_features[:n_compare]]
        
        axes[0, 1].bar(x_pos - width, centroid_scores, width, label='Centroid', color='#b39ddb')
        axes[0, 1].bar(x_pos, anova_scores, width, label='ANOVA', color='#4c3fbd')
        axes[0, 1].bar(x_pos + width, rf_scores, width, label='Random Forest', color='#7e57c2')
        
        axes[0, 1].set_xlabel('Feature Rank', fontsize=11)
        axes[0, 1].set_ylabel('Importance Score', fontsize=11)
        axes[0, 1].set_title('Feature Importance by Method (Top 10)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Cluster centroids heatmap (top 20 features, all clusters)
        n_heatmap = min(20, len(feature_scores))
        top_indices = [feature_names.index(fs['feature']) for fs in feature_scores[:n_heatmap]]
        top_names = [fs['feature'] for fs in feature_scores[:n_heatmap]]
        
        # Compute centroids for all clusters
        centroids = np.zeros((n_clusters, len(top_indices)))
        for c in range(n_clusters):
            cluster_mask = cluster_labels == c
            centroids[c, :] = np.mean(X_scaled[cluster_mask][:, top_indices], axis=0)
        
        im = axes[1, 0].imshow(centroids.T, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        axes[1, 0].set_xticks(range(n_clusters))
        axes[1, 0].set_xticklabels([f'C{i}' for i in range(n_clusters)], fontsize=10)
        axes[1, 0].set_yticks(range(len(top_names)))
        axes[1, 0].set_yticklabels(top_names, fontsize=8)
        axes[1, 0].set_xlabel('Cluster', fontsize=11)
        axes[1, 0].set_ylabel('Feature', fontsize=11)
        axes[1, 0].set_title(f'Cluster Centroids (Top {n_heatmap} Features)', fontsize=12, fontweight='bold')
        
        # Highlight best cluster
        axes[1, 0].axvline(best_cluster + 0.5, color='gold', linewidth=3, linestyle='--', alpha=0.8)
        axes[1, 0].axvline(best_cluster - 0.5, color='gold', linewidth=3, linestyle='--', alpha=0.8)
        
        plt.colorbar(im, ax=axes[1, 0], label='Standardized Value')
        
        # Plot 4: Feature group analysis (if features have prefixes)
        feature_groups = {}
        for fs in feature_scores:
            # Extract prefix (e.g., "PETdynamic" from "PETdynamic_t0")
            parts = fs['feature'].split('_')
            if len(parts) > 1:
                group = parts[0]
            else:
                group = fs['feature']
            
            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(fs['avg_importance'])
        
        # Aggregate importance by group
        group_importance = {g: np.sum(scores) for g, scores in feature_groups.items()}
        group_importance = dict(sorted(group_importance.items(), key=lambda x: x[1], reverse=True))
        
        n_groups = min(15, len(group_importance))
        groups = list(group_importance.keys())[:n_groups]
        group_scores = [group_importance[g] for g in groups]
        
        axes[1, 1].barh(range(len(groups)), group_scores, color='#7e57c2')
        axes[1, 1].set_yticks(range(len(groups)))
        axes[1, 1].set_yticklabels(groups, fontsize=9)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlabel('Total Importance Score', fontsize=11)
        axes[1, 1].set_title(f'Feature Modality Importance\n(Summed over all features)', 
                             fontsize=12, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
        plt.close()
    
    # Return results as structured dict for Hydra logging
    return {
        'top_features': [
            {
                'rank': rank,
                'feature': fs['feature'],
                'avg_importance': fs['avg_importance'],
                'centroid_importance': fs['centroid_importance'],
                'anova_importance': fs['anova_importance'],
                'rf_importance': fs['rf_importance']
            }
            for rank, fs in enumerate(feature_scores[:top_n], 1)
        ],
        'modality_importance': group_importance
    }