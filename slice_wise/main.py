"""Main script for slice-wise tumor grade classification with cross-validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ..shared.train import train
from ..shared.evaluate import (
    compute_metrics, evaluate_estimator,
    summarise_cv_folds, log_cv_summary, build_cv_results,
    estimate_trainable_parameters, compact_metrics_for_log,
)
from ..shared.utils import set_determinism, save_results
from ..shared.data_loaders import get_loader
from ..shared.data import make_cv_folds
from ..shared.aggregate import aggregate_patients

# Local slice-wise imports
from .data import build_slice_batch, build_slice_index
from .model import ModularTumorClassifier


logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute the slice-wise tumor grade classification pipeline with cross-validation."""
    
    # Setup
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Output directory: {output_dir}")
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "Data"
    
    set_determinism(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    logger.info(f"Loading {dataset_name} dataset...")
    df = get_loader(dataset_name, dataset_config, DATA_DIR).load()

    n_folds = cfg.data.n_folds
    logger.info(f"\n{'='*70}")
    logger.info(f"USING {n_folds}-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*70}\n")

    labels_all, patient_ids_all, slice_metadata_all, feature_names, slice_info = build_slice_index(
        df=df,
        feature_cols=cfg.features[dataset_name].feature_cols,
        planes=cfg.features.planes,
        min_tumor_pixels=cfg.features.min_tumor_pixels,
        tumor_only=cfg.features.tumor_only,
    )
    patient_grades = df['grade'].values.astype(int)
    
    logger.info(f"Total slices: {slice_info['n_slices_total']}")
    logger.info(f"Total channels: {slice_info['n_channels']}")
    logger.info(f"Patients: {slice_info['n_patients']} for {n_folds}-fold CV\n")
    
    # Determine actual number of input channels (after PET_dynamic expansion)
    n_input_channels = slice_info['n_channels']
    logger.info(f"Actual input channels: {n_input_channels}\n")
    
    # Phase 2 is optional; default to disabled when section is absent.
    phase2_checkpoint = None
    unfreeze_n_layers = 0
    phase2_lr = 1e-5
    phase2_epochs = 10
    phase2_batch_size = 16
    if 'phase2' in cfg and cfg.phase2 is not None:
        if cfg.phase2.get('checkpoint_path') is not None:
            phase2_checkpoint = cfg.phase2.checkpoint_path
        unfreeze_n_layers = int(cfg.phase2.get('unfreeze_n_layers', 0))
        phase2_lr = float(cfg.phase2.get('learning_rate', phase2_lr))
        phase2_epochs = int(cfg.phase2.get('num_epochs', phase2_epochs))
        phase2_batch_size = int(cfg.phase2.get('batch_size', phase2_batch_size))

    if phase2_checkpoint:
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 2: Fine-tuning from Phase 1 checkpoint")
        logger.info(f"{'='*70}")
        logger.info(f"Checkpoint directory: {Path(phase2_checkpoint).parent}")
        logger.info(f"Unfreezing last {unfreeze_n_layers} backbone block(s)")
        logger.info(f"lr={phase2_lr}, epochs={phase2_epochs}, batch_size={phase2_batch_size}\n")
    
    # Get list of classifier types to test
    classifier_types = cfg.model.classifier.types if hasattr(cfg.model.classifier, 'types') else [cfg.model.classifier.type]
    logger.info(f"Testing {len(classifier_types)} classifier type(s): {classifier_types}\n")
    
    # Store results for all classifiers
    all_classifier_results = {}
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for classifier_type in classifier_types:
        logger.info(f"\n{'='*70}")
        logger.info(f"CLASSIFIER: {classifier_type.upper()}")
        logger.info(f"{'='*70}\n")
        
        fold_results = []

        for fold_idx, (fold_train_patients, fold_test_patients) in enumerate(
            make_cv_folds(patient_ids_all, patient_grades, n_folds, cfg.data.random_state)
        ):
            logger.info(f"\n{'='*70}")
            logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
            logger.info(f"{'='*70}\n")
            
            train_mask = np.isin(patient_ids_all, fold_train_patients)
            test_mask = np.isin(patient_ids_all, fold_test_patients)
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            y_train, train_pids = labels_all[train_mask], patient_ids_all[train_mask]
            y_test, test_pids = labels_all[test_mask], patient_ids_all[test_mask]
            
            logger.info(f"Fold {fold_idx + 1}: {len(train_indices)} train slices from {len(fold_train_patients)} patients")
            logger.info(f"Fold {fold_idx + 1}: {len(test_indices)} test slices from {len(fold_test_patients)} patients")
            
            # Log class distribution for this fold
            train_patient_grades = patient_grades[fold_train_patients]
            test_patient_grades = patient_grades[fold_test_patients]
            logger.info(f"  Train patients: LGG={np.sum(train_patient_grades==0)}, HGG={np.sum(train_patient_grades==1)}")
            logger.info(f"  Test patients: LGG={np.sum(test_patient_grades==0)}, HGG={np.sum(test_patient_grades==1)}")
            
            # Create model
            assert cfg.model.type == 'modular_cnn', (
                f"Unsupported slice-wise model type '{cfg.model.type}'. Expected 'modular_cnn'."
            )
            classifier_config = OmegaConf.to_container(cfg.model.classifier[classifier_type], resolve=True)
            model_config = OmegaConf.to_container(cfg.model.modular_cnn, resolve=True)
            model_config['classifier_type'] = classifier_type
            model_config.update(classifier_config)
            model_config['n_input_channels'] = n_input_channels
            
            model = ModularTumorClassifier(**model_config).to(device)
            n_trainable_torch_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            n_total_torch_params = int(sum(p.numel() for p in model.parameters()))
            
            batch_size = cfg.train.inference_batch_size

            def _build_slice_batch_from_global_indices(global_indices: np.ndarray) -> np.ndarray:
                batch_metadata = slice_metadata_all[global_indices]
                return build_slice_batch(
                    df=df,
                    feature_cols=cfg.features[dataset_name].feature_cols,
                    metadata_batch=batch_metadata,
                    n_channels=n_input_channels,
                    target_size=cfg.features.target_size,
                    tumor_only=cfg.features.tumor_only,
                )

            if fold_idx in feature_cache:
                X_train_features, X_test_features = feature_cache[fold_idx]
                logger.info(
                    f"Using cached CNN features for fold {fold_idx + 1} | "
                    f"train={X_train_features.shape}, test={X_test_features.shape}"
                )
            else:
                # Phase 2: load Phase 1 weights and fine-tune end-to-end before extracting features.
                if phase2_checkpoint:
                    model.load_phase1_checkpoint(
                        checkpoint_dir=Path(phase2_checkpoint).parent,
                        fold_idx=fold_idx,
                        expected_n_channels=n_input_channels,
                        device=device,
                        unfreeze_n_layers=unfreeze_n_layers,
                    )
                    logger.info(f"Loaded Phase 1 checkpoint for fold {fold_idx + 1}")
                    if unfreeze_n_layers > 0:
                        logger.info(
                            f"Fine-tuning fold {fold_idx + 1} "
                            f"(lr={phase2_lr}, epochs={phase2_epochs}, bs={phase2_batch_size})..."
                        )

                        def _build_batch_for_train(local_idx: np.ndarray) -> np.ndarray:
                            return _build_slice_batch_from_global_indices(train_indices[local_idx])

                        ft_losses = model.fine_tune(
                            build_batch=_build_batch_for_train,
                            y=y_train,
                            n_samples=len(train_indices),
                            device=device,
                            lr=phase2_lr,
                            num_epochs=phase2_epochs,
                            batch_size=phase2_batch_size,
                            seed=cfg.train.seed,
                        )
                        logger.info(
                            f"Fold {fold_idx + 1} fine-tune losses (per epoch): "
                            f"{[round(l, 4) for l in ft_losses]}"
                        )

                logger.info(f"Extracting CNN features for fold {fold_idx + 1} (cached for all classifiers)...")

                def _extract_features_from_indices(indices: np.ndarray) -> np.ndarray:
                    features_list = []
                    model.eval()
                    with torch.no_grad():
                        for i in range(0, len(indices), batch_size):
                            batch_indices = indices[i : i + batch_size]
                            batch_slices = _build_slice_batch_from_global_indices(batch_indices)
                            batch_tensor = torch.from_numpy(batch_slices).to(device)
                            batch_features = model.backbone(model.adapter(batch_tensor)).cpu().numpy()
                            features_list.append(batch_features)
                            del batch_slices, batch_tensor, batch_features
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    return np.vstack(features_list)

                X_train_features = _extract_features_from_indices(train_indices)
                X_test_features = _extract_features_from_indices(test_indices)

                # Per-fold feature-space z-score normalisation (fit on training only).
                means = np.mean(X_train_features, axis=0, keepdims=True)
                stds = np.std(X_train_features, axis=0, keepdims=True)
                stds[stds == 0] = 1.0
                X_train_features = (X_train_features - means) / stds
                X_test_features = (X_test_features - means) / stds

                feature_cache[fold_idx] = (X_train_features, X_test_features)
                logger.info(f"Extracted features shape: {X_train_features.shape}")

            train(model.classifier, X_train_features, y_train, device)
            n_trainable_params_estimate = estimate_trainable_parameters(model)

            _, train_probs, train_slice_metrics = evaluate_estimator(
                model.classifier,
                X_train_features,
                y_train,
                device,
                batch_size=batch_size,
                apply_sigmoid=False,
            )
            _, test_probs, test_slice_metrics = evaluate_estimator(
                model.classifier,
                X_test_features,
                y_test,
                device,
                batch_size=batch_size,
                apply_sigmoid=False,
            )

            history = None

            # ========== AGGREGATE TO PATIENT LEVEL ==========
            (train_pred_patient, train_label_patient, train_prob_patient), \
            (test_pred_patient,  test_label_patient,  test_prob_patient) = aggregate_patients(
                method=cfg.patient_aggregation,
                train_prob=train_probs, train_true=y_train, train_pids=train_pids,
                test_prob=test_probs,  test_true=y_test,  test_pids=test_pids,
                mil_epochs=cfg.mil.epochs,
                mil_lr=cfg.mil.lr,
            )
                
            train_patient_metrics = compute_metrics(train_label_patient, train_pred_patient, train_prob_patient)
            test_patient_metrics = compute_metrics(test_label_patient, test_pred_patient, test_prob_patient)
            
            classifier_dir = os.path.join(output_dir, classifier_type)
            os.makedirs(classifier_dir, exist_ok=True)
            # Save the adapter + backbone checkpoint for the first classifier in each
            # fold (the cached fine-tuned weights are captured only on that pass).
            # Subsequent classifiers in the same fold share those weights via the cache.
            is_first_classifier_for_fold = (classifier_type == classifier_types[0])
            if is_first_classifier_for_fold:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_fold_{fold_idx + 1}.pth')
                torch.save({
                    'fold': fold_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'n_input_channels': n_input_channels,
                    'phase': 2 if phase2_checkpoint else 1,
                    'source_checkpoint': str(phase2_checkpoint) if phase2_checkpoint else None,
                }, checkpoint_path)
                logger.info(
                    f"Saved {'Phase 2 (fine-tuned)' if phase2_checkpoint else 'Phase 1'} "
                    f"checkpoint: {checkpoint_path}"
                )
            
            # Log results
            logger.info(f"\nFold {fold_idx + 1} Results:")
            logger.info(f"  Slice-level train: {compact_metrics_for_log(train_slice_metrics)}")
            logger.info(f"  Slice-level test:  {compact_metrics_for_log(test_slice_metrics)}")
            logger.info(f"  Patient-level train: {compact_metrics_for_log(train_patient_metrics)}")
            logger.info(f"  Patient-level test:  {compact_metrics_for_log(test_patient_metrics)}")
            
            # Store fold results
            fold_results.append({
                'fold':        fold_idx + 1,
                'n_train':     int(len(fold_train_patients)),
                'n_test':      int(len(fold_test_patients)),
                'train':       train_patient_metrics,
                'test':        test_patient_metrics,
                'train_slice': train_slice_metrics,
                'test_slice':  test_slice_metrics,
                'history':     history,
                'n_trainable_params_estimate': n_trainable_params_estimate,
                'n_trainable_torch_params': n_trainable_torch_params,
                'n_total_torch_params': n_total_torch_params,
            })
            
            # Explicit garbage collection to prevent OOM
            del X_train_features, X_test_features
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Explicitly clear Keras/TF backend if using path_foundation
            if cfg.model.modular_cnn.backbone == 'path_foundation':
                try:
                    import keras
                    keras.backend.clear_session()
                except ImportError:
                    pass
        
        # Aggregate results for this classifier
        summary = summarise_cv_folds(fold_results)
        log_cv_summary(logger, summary)
        results = build_cv_results(
            fold_results=fold_results,
            summary=summary,
            config=OmegaConf.to_container(cfg, resolve=True),
            granularity='slice-wise',
            n_input_channels=n_input_channels,
            feature_names=feature_names,
        )
        classifier_dir = os.path.join(output_dir, classifier_type)
        os.makedirs(classifier_dir, exist_ok=True)
        save_results(results, os.path.join(classifier_dir, 'results.json'))
        logger.info(f"\n✓ Results for {classifier_type} saved to {classifier_dir}")
        
        all_classifier_results[classifier_type] = results
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL CLASSIFIERS COMPLETED")
    logger.info(f"{'='*70}\n")
    logger.info(f"✓ All results saved to {output_dir}")


if __name__ == "__main__":
    main()