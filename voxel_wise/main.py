"""Main script for voxel-wise tumor grade classification with cross-validation."""

from __future__ import annotations

import os
import logging
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# Package imports from shared
from ..shared.model import create_model
from ..shared.train import train
from ..shared.evaluate import (
    compute_metrics, evaluate_estimator,
    summarise_cv_folds, log_cv_summary, build_cv_results, estimate_trainable_parameters,
)
from ..shared.utils import set_determinism, save_results


from ..shared.data_loaders import get_loader
from ..shared.data import make_cv_folds
from ..shared.aggregate import aggregate_patients
from .data import extract_voxel_features, scale_fold_split

from .hybrid_clustering.hybrid_predictor import HybridClusterPredictor

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute the voxel-wise tumor grade classification pipeline with CV."""
    
    # Setup
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Output directory: {output_dir}")
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "Data"
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    set_determinism(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    model_type = cfg.model.type
    
    # Load data
    logger.info(f"Loading {dataset_name} dataset...")
    df = get_loader(dataset_name, dataset_config, DATA_DIR).load()

    n_folds = cfg.data.n_folds
    logger.info(f"\n{'='*70}")
    logger.info(f"USING {n_folds}-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*70}\n")
    
    X_all, y_all, patient_ids, feature_names = extract_voxel_features(
        df=df,
        feature_cols=cfg.features[dataset_name].feature_cols,
        sample_ratio=cfg.features.sample_ratio,
        max_voxels_per_patient=cfg.features.get('max_voxels_per_patient', None),
        tumor_only=cfg.features.tumor_only,
    )
    patient_grades = df['grade'].values.astype(int)

    feature_info = {
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'n_voxels_total': len(X_all),
        'n_patients': int(len(np.unique(patient_ids))),
        'n_folds': n_folds,
        'class_distribution': {
            'low_grade': int(np.sum(patient_grades == 0)),
            'high_grade': int(np.sum(patient_grades == 1)),
            'total': len(patient_grades),
        },
        'voxels_per_patient_avg': len(X_all) / len(df),
        'sample_ratio': cfg.features.sample_ratio,
    }

    logger.info(f"Total features: {feature_info['n_features']}")
    logger.info(f"Total voxels: {feature_info['n_voxels_total']}")
    logger.info(f"Patients: {feature_info['n_patients']} for {n_folds}-fold CV\n")

    fold_results = []

    for fold_idx, (fold_train_patients, fold_test_patients) in enumerate(
        make_cv_folds(patient_ids, patient_grades, n_folds, cfg.data.random_state)
    ):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
        logger.info(f"{'='*70}\n")

        # Get fold data with proper scaling
        X_train, y_train, train_pids, X_test, y_test, test_pids, scaler = scale_fold_split(
            X_all, y_all, patient_ids, fold_train_patients, fold_test_patients
        )
        
        logger.info(f"Fold {fold_idx + 1}: {len(X_train)} train voxels from {len(fold_train_patients)} patients")
        logger.info(f"Fold {fold_idx + 1}: {len(X_test)} test voxels from {len(fold_test_patients)} patients")
        
        use_hybrid = cfg.hybrid_clustering.enabled
        
        if use_hybrid:
            # ========== HYBRID CLUSTER-FILTERED APPROACH ==========
            logger.info("\n→ Using Hybrid Cluster-Filtered Classification")
            
            hybrid_config = cfg.hybrid_clustering
            
            # Create base classifier (unwrapped)
            model_config = OmegaConf.to_container(cfg.model[model_type], resolve=True)
            base_model = create_model(model_type=model_type, **model_config)
            if isinstance(base_model, torch.nn.Module):
                base_model = base_model.to(device)
            
            # Wrap in hybrid predictor
            model = HybridClusterPredictor(
                base_classifier=base_model,
                device=device,
                n_clusters=hybrid_config.n_clusters,
                method=hybrid_config.method,
                criterion=hybrid_config.criterion,
                min_cluster_voxels=hybrid_config.min_cluster_voxels,
                random_state=cfg.train.seed,
            )

            model.fit(X_train, y_train, train_pids)
            train_prob_voxel = model.predict_proba(X_train)
            test_prob_voxel  = model.predict_proba(X_test)
            cluster_info = model.get_cluster_info()
            n_trainable_params_estimate = estimate_trainable_parameters(model)
            
        else:
            # ========== STANDARD APPROACH (NO CLUSTERING) ==========
            model_config = OmegaConf.to_container(cfg.model[model_type], resolve=True)
            model = create_model(model_type=model_type, **model_config)
            if isinstance(model, torch.nn.Module):
                model = model.to(device)
            train(model, X_train, y_train, device)

            batch_size = 10000
            _, train_prob_voxel, _ = evaluate_estimator(model, X_train, y_train, device, batch_size=batch_size)
            _, test_prob_voxel,  _ = evaluate_estimator(model, X_test,  y_test,  device, batch_size=batch_size)

            cluster_info = None
            n_trainable_params_estimate = estimate_trainable_parameters(model)
        
        # ========== AGGREGATE TO PATIENT LEVEL ==========
        (train_pred_patient, train_label_patient, train_prob_patient), \
        (test_pred_patient,  test_label_patient,  test_prob_patient) = aggregate_patients(
            method=cfg.patient_aggregation,
            train_prob=train_prob_voxel, train_true=y_train, train_pids=train_pids,
            test_prob=test_prob_voxel,   test_true=y_test,  test_pids=test_pids,
            mil_epochs=cfg.mil.epochs,
            mil_lr=cfg.mil.lr,
        )
        train_metrics = compute_metrics(train_label_patient, train_pred_patient, train_prob_patient)
        test_metrics = compute_metrics(test_label_patient, test_pred_patient, test_prob_patient)

        logger.info(f"\nFold {fold_idx + 1} Results (Patient-level):")
        logger.info(f"  Train: Acc={train_metrics['accuracy']:.4f}, Bal_Acc={train_metrics['balanced_accuracy']:.4f}, MCC={train_metrics['mcc']:.4f}, F1={train_metrics['f1_macro']:.4f}")
        logger.info(f"  Test:  Acc={test_metrics['accuracy']:.4f}, Bal_Acc={test_metrics['balanced_accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, F1={test_metrics['f1_macro']:.4f}")

        fold_results.append({
            'fold':    fold_idx + 1,
            'n_train': int(len(fold_train_patients)),
            'n_test':  int(len(fold_test_patients)),
            'train':   train_metrics,
            'test':    test_metrics,
            'n_trainable_params_estimate': n_trainable_params_estimate,
            **({'cluster_info': cluster_info} if use_hybrid else {}),
        })
    
    # CV Summary
    summary = summarise_cv_folds(fold_results)
    log_cv_summary(logger, summary)
    results = build_cv_results(
        fold_results=fold_results,
        summary=summary,
        config=OmegaConf.to_container(cfg, resolve=True),
        granularity='voxel-wise',
        model_type=model_type,
        feature_info=feature_info,
    )
    results_path = os.path.join(output_dir, 'results.json')
    save_results(results, results_path)
    logger.info(f"\n✓ Results saved to: {results_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info("VOXEL-WISE CV PIPELINE COMPLETED")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()