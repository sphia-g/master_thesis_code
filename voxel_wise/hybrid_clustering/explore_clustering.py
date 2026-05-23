"""
Interactive exploration of clustering on Naive40 dataset.
Run this ONCE to understand clustering behavior before using the automated pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf

# Add parent directories to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from Classification.shared.data_loaders import get_loader
from Classification.voxel_wise.data import extract_voxel_features
from Classification.voxel_wise.hybrid_clustering.clustering import (
    find_optimal_clusters, fit_clusterer, get_discriminative_cluster
)
from Classification.voxel_wise.hybrid_clustering.analysis import (
    plot_cluster_metrics, plot_cluster_separation, plot_patient_cluster_heatmap,
    plot_patient_cluster_bars, analyze_cluster_feature_importance
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path setup
CONFIG_PATH = PROJECT_ROOT / "Code/Classification/voxel_wise/config.yaml"
DATA_ROOT = PROJECT_ROOT / "Data"

def main():
    """Explore clustering on chosen dataset."""
    
    logger.info("="*70)
    logger.info(f"CLUSTERING EXPLORATION")
    logger.info("="*70)
    
    # Load config
    cfg = OmegaConf.load(CONFIG_PATH)
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    
    # Load data
    logger.info(f"\n1. Loading {dataset_name} dataset...")
    df = get_loader(dataset_name, dataset_config, DATA_ROOT).load()
    logger.info(f"   Loaded {len(df)} patients")
    
    # Create dataset-specific output directory
    OUTPUT_DIR = Path(__file__).parent / "outputs" / f"exploration_{dataset_name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Extract voxel features
    logger.info(f"\n2. Extracting tumor voxel features...")
    feature_cols = cfg.features[dataset_name]['feature_cols']
    sample_ratio = cfg.features.sample_ratio
    max_voxels = cfg.features.get('max_voxels_per_patient', None)
    
    X, y, patient_ids, feature_names = extract_voxel_features(
        df=df,
        feature_cols=feature_cols,
        sample_ratio=sample_ratio,
        max_voxels_per_patient=max_voxels,
        tumor_only=True,
    )
    
    logger.info(f"\n   Feature matrix: {X.shape}")
    logger.info(f"   LGG voxels: {np.sum(y==0):,} ({np.sum(y==0)/len(y)*100:.1f}%)")
    logger.info(f"   HGG voxels: {np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.1f}%)")
    logger.info(f"   Patients: {len(np.unique(patient_ids))}")
    
    # Scale features
    # shouldnt scaling be already done when data come from data.py?
    logger.info(f"\n3. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"   Features standardized (mean=0, std=1)")
    
    # Find optimal k
    logger.info(f"\n4. Finding optimal number of clusters...")
    k_range = range(2, 11)
    method = "kmeans"  # or "gmm"
    
    metrics, best_k = find_optimal_clusters(
        X=X_scaled,
        k_range=k_range,
        method=method,
        random_state=42
    )
    
    # Plot cluster metrics
    logger.info(f"\n5. Plotting cluster selection metrics...")
    plot_cluster_metrics(
        metrics=metrics,
        save_path=OUTPUT_DIR / f"{dataset_name}_cluster_metrics.png",
        title_suffix="Full Dataset"
    )
    
    # Test multiple k values
    logger.info(f"\n6. Analyzing LGG/HGG separation for different k values...")
    logger.info("="*70)
    
    k_to_test = [best_k['bic'], best_k['silhouette']]  # Test BIC and Silhouette optima
    k_to_test = list(range(2, 8))
    
    results_summary = []
    
    for k in k_to_test:
        logger.info(f"\n--- Testing k={k} ---")
        
        # Fit clusterer
        clusterer = fit_clusterer(X_scaled, n_clusters=k, method=method, random_state=42)
        cluster_labels = clusterer.predict(X_scaled)
        
        # Find discriminative cluster
        best_cluster, cluster_stats = get_discriminative_cluster(
            cluster_labels=cluster_labels,
            y=y,
            patient_ids=patient_ids,
            criterion="percentage_diff"
        )
        
        # Plot separation
        plot_cluster_separation(
            cluster_stats=cluster_stats,
            selected_cluster=best_cluster,
            save_path=OUTPUT_DIR / f"{dataset_name}_cluster_separation_k{k}.png",
            title_suffix=f"k={k}, Full Dataset"
        )
        
        # Plot patient-level heatmap
        plot_patient_cluster_heatmap(
            cluster_labels=cluster_labels,
            y=y,
            patient_ids=patient_ids,
            n_clusters=k,
            save_path=OUTPUT_DIR / f"{dataset_name}_patient_heatmap_k{k}.png",
            title_suffix=f"k={k}"
        )

        # Plot patient-level bar charts
        plot_patient_cluster_bars(
            cluster_labels=cluster_labels,
            y=y,
            patient_ids=patient_ids,
            n_clusters=k,
            save_path=OUTPUT_DIR / f"{dataset_name}_patient_bars_k{k}.png",
            title_suffix=f"k={k}"
        )
        
        # Summary stats
        n_selected_voxels = cluster_stats[best_cluster]['n_voxels']
        pct_retained = n_selected_voxels / len(X) * 100
        separation = cluster_stats[best_cluster]['percentage_diff']
        
        results_summary.append({
            'k': k,
            'best_cluster': best_cluster,
            'separation': separation,
            'n_voxels': n_selected_voxels,
            'pct_retained': pct_retained
        })
        
        logger.info(f"\n   Summary for k={k}:")
        logger.info(f"   - Best cluster: {best_cluster}")
        logger.info(f"   - Separation: {separation:.1f}%")
        logger.info(f"   - Voxels retained: {n_selected_voxels:,} ({pct_retained:.1f}%)")
        
        # ============================================================
        # STATISTICAL ANALYSIS - ALL CLUSTERS
        # I truly dont like the statistical analysis here. maybe comment this out entirely? 
        # ============================================================
        from scipy.stats import mannwhitneyu, shapiro
        
        logger.info(f"\n   Statistical Analysis: Testing ALL {k} clusters")
        logger.info(f"   {'='*70}")

        # Get all unique patients (needed for all tests)
        all_unique_patients = np.unique(patient_ids)
        all_patient_labels = np.array([y[patient_ids == p][0] for p in all_unique_patients])
        total_lgg = np.sum(all_patient_labels == 0)
        total_hgg = np.sum(all_patient_labels == 1)
        
        cluster_test_results = []
        
        # TEST EVERY CLUSTER (not just the "best" one)
        for cluster_id in range(k):
            logger.info(f"\n   --- Cluster {cluster_id} ---")
            
            # Calculate patient-level percentages for THIS cluster
            lgg_percentages = []
            hgg_percentages = []
            
            for patient in all_unique_patients:
                patient_mask = patient_ids == patient
                patient_cluster_mask = (cluster_labels == cluster_id) & patient_mask
                
                # Percentage of this patient's voxels in the cluster
                n_patient_voxels = np.sum(patient_mask)
                n_cluster_voxels = np.sum(patient_cluster_mask)
                percentage = (n_cluster_voxels / n_patient_voxels * 100) if n_patient_voxels > 0 else 0.0
                
                patient_label = y[patient_ids == patient][0]
                if patient_label == 0:
                    lgg_percentages.append(percentage)
                else:
                    hgg_percentages.append(percentage)
            
            lgg_percentages = np.array(lgg_percentages)
            hgg_percentages = np.array(hgg_percentages)
            
            # Descriptive statistics
            lgg_median = np.median(lgg_percentages)
            lgg_mean = np.mean(lgg_percentages)
            lgg_std = np.std(lgg_percentages)
            hgg_median = np.median(hgg_percentages)
            hgg_mean = np.mean(hgg_percentages)
            hgg_std = np.std(hgg_percentages)
            diff_median = hgg_median - lgg_median
            
            logger.info(f"   Tumor coverage in cluster (% of patient's tumor):")
            logger.info(f"     LGG: median={lgg_median:.2f}%, mean={lgg_mean:.2f}%, std={lgg_std:.2f}%")
            logger.info(f"     HGG: median={hgg_median:.2f}%, mean={hgg_mean:.2f}%, std={hgg_std:.2f}%")
            logger.info(f"     Difference (HGG - LGG): {diff_median:+.2f}%")
            
            # Mann-Whitney U test (two-sided to test ANY difference)
            n_lgg = len(lgg_percentages)
            n_hgg = len(hgg_percentages)
            
            if n_lgg > 0 and n_hgg > 0:
                statistic, p_value = mannwhitneyu(hgg_percentages, lgg_percentages, alternative='two-sided')
                
                # Calculate effect size (rank-biserial correlation)
                # Common Language Effect Size: probability that random HGG > random LGG
                cles = statistic / (n_lgg * n_hgg)
                rank_biserial = 2 * cles - 1
                
                # Ensure sign matches the medians (positive = HGG > LGG)
                if hgg_median < lgg_median:
                    rank_biserial = -rank_biserial
                
                logger.info(f"   Mann-Whitney U test:")
                logger.info(f"     U statistic: {statistic:.1f}")
                logger.info(f"     p-value: {p_value:.4f}")
                logger.info(f"     Effect size (rank-biserial r): {rank_biserial:+.3f}")
                
                # Interpret effect size
                abs_r = abs(rank_biserial)
                if abs_r < 0.1:
                    effect_interp = "negligible"
                elif abs_r < 0.3:
                    effect_interp = "small"
                elif abs_r < 0.5:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
                
                # Direction: simply compare medians
                if hgg_median > lgg_median:
                    direction = "HGG > LGG"
                elif hgg_median < lgg_median:
                    direction = "LGG > HGG"
                else:
                    direction = "no difference"
                
                logger.info(f"     Interpretation: {effect_interp} effect ({direction})")
                
                if p_value < 0.05:
                    logger.info(f"     ✓ Statistically significant (p < 0.05)")
                else:
                    logger.info(f"     ✗ Not significant (p ≥ 0.05)")
            else:
                statistic, p_value, rank_biserial = np.nan, np.nan, np.nan
                logger.info(f"     ⚠️  Insufficient data for test")
            
            # Store results
            cluster_test_results.append({
                'cluster_id': cluster_id,
                'lgg_median': lgg_median,
                'lgg_mean': lgg_mean,
                'lgg_std': lgg_std,
                'hgg_median': hgg_median,
                'hgg_mean': hgg_mean,
                'hgg_std': hgg_std,
                'diff_median': diff_median,
                'u_statistic': statistic,
                'p_value': p_value,
                'rank_biserial': rank_biserial,
                'n_lgg_patients': n_lgg,
                'n_hgg_patients': n_hgg,
                'n_voxels': np.sum(cluster_labels == cluster_id),
                'pct_voxels': np.sum(cluster_labels == cluster_id) / len(cluster_labels) * 100
            })
        
        # ============================================================
        # COMPARE ALL CLUSTERS
        # ============================================================
        logger.info(f"\n   {'='*70}")
        logger.info(f"   COMPARISON: All {k} clusters for k={k}")
        logger.info(f"   {'='*70}")
        logger.info(f"   {'Cluster':<8} {'LGG%':<10} {'HGG%':<10} {'Diff':<10} "
                   f"{'p-value':<10} {'Effect(r)':<12} {'Sig':<5}")
        logger.info(f"   {'-'*75}")
        
        for res in cluster_test_results:
            sig_marker = "✓" if res['p_value'] < 0.05 else "✗"
            logger.info(f"   {res['cluster_id']:<8} "
                       f"{res['lgg_median']:<10.2f} "
                       f"{res['hgg_median']:<10.2f} "
                       f"{res['diff_median']:<+10.2f} "
                       f"{res['p_value']:<10.4f} "
                       f"{res['rank_biserial']:<+12.3f} "
                       f"{sig_marker:<5}")
        
        # Find cluster with strongest effect
        valid_results = [r for r in cluster_test_results if not np.isnan(r['p_value'])]
        
        if valid_results:
            most_significant = min(valid_results, key=lambda x: x['p_value'])
            largest_effect = max(valid_results, key=lambda x: abs(x['rank_biserial']))
            
            logger.info(f"\n   Key findings:")
            logger.info(f"     → Most significant: Cluster {most_significant['cluster_id']} "
                       f"(p={most_significant['p_value']:.4f})")
            logger.info(f"     → Largest effect: Cluster {largest_effect['cluster_id']} "
                       f"(r={largest_effect['rank_biserial']:+.3f}, "
                       f"|r|={abs(largest_effect['rank_biserial']):.3f})")
            
            # Check if "best_cluster" matches most discriminative
            if best_cluster == largest_effect['cluster_id']:
                logger.info(f"     ✓ Largest effect matches selected cluster ({best_cluster})")
            else:
                logger.info(f"     ⚠️  Selected cluster ({best_cluster}) differs from "
                           f"largest effect ({largest_effect['cluster_id']})")
            
            # Multiple testing correction (Bonferroni)
            alpha_corrected = 0.05 / k
            logger.info(f"\n   Multiple testing correction (Bonferroni):")
            logger.info(f"     Adjusted α = 0.05 / {k} = {alpha_corrected:.4f}")
            
            bonferroni_survivors = [r for r in valid_results if r['p_value'] < alpha_corrected]
            
            if bonferroni_survivors:
                logger.info(f"     Clusters surviving correction:")
                for res in bonferroni_survivors:
                    logger.info(f"       Cluster {res['cluster_id']}: "
                               f"p={res['p_value']:.4f}, r={res['rank_biserial']:+.3f}")
            else:
                logger.info(f"     ⚠️  No clusters survive Bonferroni correction")
        
        # Store comprehensive results
        results_summary[-1]['cluster_test_results'] = cluster_test_results
        results_summary[-1]['most_significant_cluster'] = most_significant['cluster_id'] if valid_results else None
        results_summary[-1]['largest_effect_cluster'] = largest_effect['cluster_id'] if valid_results else None
            
        # ============================================================
        # FEATURE IMPORTANCE ANALYSIS
        # ============================================================
        feature_importance = analyze_cluster_feature_importance(
            X_scaled=X_scaled,
            cluster_labels=cluster_labels,
            feature_names=feature_names,
            best_cluster=best_cluster,
            k=k,
            save_path=OUTPUT_DIR / f"{dataset_name}_feature_importance_k{k}.png",
            title_suffix=f"k={k}, {dataset_name}",
            top_n=20
        )
        
        # Store for summary (optional)
        results_summary[-1]['top_features'] = feature_importance['top_features'][:5]  # Top 5 only
            
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("EXPLORATION SUMMARY")
    logger.info(f"{'='*70}\n")
    
    logger.info(f"Optimal k values:")
    logger.info(f"  - AIC: {best_k['aic']}")
    logger.info(f"  - BIC: {best_k['bic']}")
    logger.info(f"  - Silhouette: {best_k['silhouette']}")
    logger.info(f"  - Davies-Bouldin: {best_k['dbi']}")
    
    logger.info(f"\nSeparation analysis:")
    for res in results_summary:
        logger.info(f"  k={res['k']}: "
                   f"Cluster {res['best_cluster']} with {res['separation']:.1f}% separation, "
                   f"{res['pct_retained']:.1f}% voxels retained")
    
    # Recommendation
    logger.info(f"\n{'='*70}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*70}\n")
    
    # Find k with best separation
    best_separation_k = max(results_summary, key=lambda x: x['separation'])
    logger.info(f"1. Best separation: k={best_separation_k['k']} "
               f"({best_separation_k['separation']:.1f}% difference)")
    
    # Find k with most voxels retained
    best_retention_k = max(results_summary, key=lambda x: x['pct_retained'])
    logger.info(f"2. Most voxels retained: k={best_retention_k['k']} "
               f"({best_retention_k['pct_retained']:.1f}%)")
    
    # Balance recommendation
    # Score = separation * sqrt(retention) to balance both
    for res in results_summary:
        res['balance_score'] = res['separation'] * np.sqrt(res['pct_retained'])
    
    best_balance_k = max(results_summary, key=lambda x: x['balance_score'])
    logger.info(f"3. Best balance (separation × √retention): k={best_balance_k['k']}")
    
    logger.info(f"\n→ Suggested configuration for main pipeline:")
    logger.info(f"   hybrid_clustering:")
    logger.info(f"     enabled: true")
    logger.info(f"     method: {method}")
    logger.info(f"     n_clusters: {best_balance_k['k']}")
    logger.info(f"     find_optimal_k: false  # or true to search per fold")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()