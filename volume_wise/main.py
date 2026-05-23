"""Main script for volume-wise 3D tumor grade classification with cross-validation."""

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
from ..shared.utils import save_results, set_determinism
from ..shared.evaluate import (
    evaluate_estimator,
    summarise_cv_folds, log_cv_summary, build_cv_results, estimate_trainable_parameters,
)
from ..shared.data_loaders import get_loader
from ..shared.data import make_cv_folds

from .data import scale_fold_split, extract_patient_volumes
from .model import VolumeWiseClassifier

logger = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute the volume-wise 3D tumor grade classification pipeline with CV."""

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Output directory: {output_dir}")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "Data"
    assert cfg.model.type in cfg.model, (
        f"Missing model config section '{cfg.model.type}' under cfg.model"
    )
    model_cfg = cfg.model[cfg.model.type]

    classifier_type = cfg.model.classifier.type
    assert classifier_type in cfg.model.classifier, (
        f"Missing classifier config section '{classifier_type}' under cfg.model.classifier"
    )
    classifier_kwargs = OmegaConf.to_container(
        cfg.model.classifier[classifier_type],
        resolve=True,
    )

    use_pretrained = bool(model_cfg.pretrained) if 'pretrained' in model_cfg else True
    PRETRAINED_PATH = str(PROJECT_ROOT / model_cfg.pretrained_path) if use_pretrained else None

    set_determinism(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if use_pretrained:
        logger.info(f"Backbone init: pretrained weights from {PRETRAINED_PATH}")
    else:
        logger.info("Backbone init: random (pretrained=false)")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    logger.info(f"Loading {dataset_name} dataset from {DATA_DIR}...")
    df = get_loader(dataset_name, dataset_config, DATA_DIR).load()

    n_folds = cfg.data.n_folds
    target_shape = tuple(cfg.features.target_shape)

    logger.info(f"\n{'='*70}")
    logger.info(f"USING {n_folds}-FOLD CROSS-VALIDATION")
    logger.info(f"Target volume shape: {target_shape}")
    logger.info(f"{'='*70}\n")

    X_all, y_all, patient_ids, _, vol_info = extract_patient_volumes(
        df=df,
        feature_cols=list(cfg.features[dataset_name].feature_cols),
        target_shape=target_shape,
        tumor_only=cfg.features.tumor_only,
    )
    patient_grades = df['grade'].values.astype(int)

    n_input_channels = X_all.shape[1]
    logger.info(f"Total patients: {len(X_all)}, Channels: {n_input_channels}\n")

    classifier_types = cfg.model.classifier.types if hasattr(cfg.model.classifier, 'types') else [cfg.model.classifier.type]
    logger.info(f"Testing {len(classifier_types)} classifier type(s): {classifier_types}\n")

    # Phase 2 is optional; default to disabled when section is absent.
    phase2_checkpoint = None
    phase2_unfreeze = 0
    phase2_lr = 1e-5
    phase2_epochs = 10
    phase2_batch_size = cfg.train.batch_size
    if 'phase2' in cfg and cfg.phase2 is not None:
        if cfg.phase2.get('checkpoint_path') is not None:
            phase2_checkpoint = cfg.phase2.checkpoint_path
        phase2_unfreeze = int(cfg.phase2.get('unfreeze_n_blocks', 0))
        phase2_lr = float(cfg.phase2.get('learning_rate', phase2_lr))
        phase2_epochs = int(cfg.phase2.get('num_epochs', phase2_epochs))
        phase2_batch_size = int(cfg.phase2.get('batch_size', phase2_batch_size))

    if phase2_checkpoint:
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 2: Fine-tuning from Phase 1 checkpoint")
        logger.info(f"{'='*70}")
        logger.info(f"Checkpoint directory: {Path(phase2_checkpoint).parent}")
        logger.info(f"Unfreezing last {phase2_unfreeze} backbone stage(s)")
        logger.info(f"lr={phase2_lr}, epochs={phase2_epochs}, batch_size={phase2_batch_size}\n")

    all_classifier_results = {}
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for classifier_type in classifier_types:
        logger.info(f"\n{'='*70}")
        logger.info(f"CLASSIFIER: {classifier_type.upper()}")
        logger.info(f"{'='*70}\n")

        classifier_kwargs = OmegaConf.to_container(
            cfg.model.classifier[classifier_type],
            resolve=True,
        )

        fold_results = []

        for fold_idx, (fold_train_patients, fold_test_patients) in enumerate(
            make_cv_folds(patient_ids, patient_grades, n_folds, cfg.data.random_state)
        ):
            logger.info(f"\n{'='*70}")
            logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
            logger.info(f"{'='*70}\n")

            X_train, y_train, X_test, y_test, _ = scale_fold_split(
                X_all, y_all, patient_ids, fold_train_patients, fold_test_patients
            )
            logger.info(f"Train: {len(X_train)} patients  |  Test: {len(X_test)} patients")

            # Build model
            model = VolumeWiseClassifier(
                n_input_channels=n_input_channels,
                backbone=f"medicalnet_{model_cfg.model_depth}",
                pretrained=use_pretrained,
                pretrained_path=PRETRAINED_PATH,
                freeze_backbone=model_cfg.freeze_backbone,
                classifier_type=classifier_type,
                **classifier_kwargs,
            ).to(device)
            n_trainable_torch_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            n_total_torch_params = int(sum(p.numel() for p in model.parameters()))

            infer_bs = cfg.train.inference_batch_size
            if fold_idx in feature_cache:
                X_train_feat, X_test_feat = feature_cache[fold_idx]
                logger.info(
                    f"Using cached backbone features for fold {fold_idx + 1} | "
                    f"train={X_train_feat.shape}, test={X_test_feat.shape}"
                )
            else:
                if phase2_checkpoint:
                    model.load_phase1_checkpoint(
                        checkpoint_dir=Path(phase2_checkpoint).parent,
                        fold_idx=fold_idx,
                        expected_n_channels=n_input_channels,
                        device=device,
                        unfreeze_n_layers=phase2_unfreeze,
                    )
                    logger.info(f"Loaded Phase 1 checkpoint for fold {fold_idx + 1}")
                    if phase2_unfreeze > 0:
                        logger.info(
                            f"Fine-tuning fold {fold_idx + 1} "
                            f"(lr={phase2_lr}, epochs={phase2_epochs}, bs={phase2_batch_size})..."
                        )
                        ft_losses = model.fine_tune(
                            X=X_train,
                            y=y_train,
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

                logger.info(f"Extracting backbone features for fold {fold_idx + 1} (cached for all classifiers)...")
                X_train_feat = model.extract_features(X_train, device, infer_bs)
                X_test_feat = model.extract_features(X_test, device, infer_bs)
                feature_cache[fold_idx] = (X_train_feat, X_test_feat)
                logger.info(f"Feature shape: {X_train_feat.shape}")

                # Save the adapter + backbone checkpoint for later runs.
                # Phase 1 → fresh frozen weights; Phase 2 → fine-tuned weights
                # (which can themselves be loaded as the source for a Phase 3 run).
                fold_ckpt_dir = output_dir
                os.makedirs(fold_ckpt_dir, exist_ok=True)
                fold_ckpt_path = os.path.join(
                    fold_ckpt_dir, f"checkpoint_fold_{fold_idx + 1}.pth"
                )
                torch.save({
                    'fold': fold_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'config': OmegaConf.to_container(cfg, resolve=True),
                    'n_input_channels': n_input_channels,
                    'phase': 2 if phase2_checkpoint else 1,
                    'source_checkpoint': str(phase2_checkpoint) if phase2_checkpoint else None,
                }, fold_ckpt_path)
                logger.info(
                    f"Saved {'Phase 2 (fine-tuned)' if phase2_checkpoint else 'Phase 1'} "
                    f"checkpoint: {fold_ckpt_path}"
                )

            train(model.classifier, X_train_feat, y_train, device)
            n_trainable_params_estimate = estimate_trainable_parameters(model)

            _, _, train_metrics = evaluate_estimator(
                model.classifier, X_train_feat, y_train, device
            )
            _, _, test_metrics = evaluate_estimator(
                model.classifier, X_test_feat, y_test, device
            )
            # --------------------------------------------------------------
            # Log  (already patient-level — no aggregation step needed)
            # --------------------------------------------------------------
            logger.info(f"\nFold {fold_idx + 1} Results (patient-level):")
            logger.info(f"  Train: Acc={train_metrics['accuracy']:.4f}, MCC={train_metrics['mcc']:.4f}")
            logger.info(f"  Test:  Acc={test_metrics['accuracy']:.4f},  MCC={test_metrics['mcc']:.4f}")
            logger.info(f"  Overfitting gap: {train_metrics['accuracy'] - test_metrics['accuracy']:+.4f}")

            fold_results.append({
                "fold":    fold_idx + 1,
                "n_train": int(len(X_train)),
                "n_test":  int(len(X_test)),
                "train":   train_metrics,
                "test":    test_metrics,
                "n_trainable_params_estimate": n_trainable_params_estimate,
                "n_trainable_torch_params": n_trainable_torch_params,
                "n_total_torch_params": n_total_torch_params,
            })

            del X_train, X_test, X_train_feat, X_test_feat, model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # CV Summary
        # ------------------------------------------------------------------
        summary = summarise_cv_folds(fold_results)
        log_cv_summary(logger, summary)
        results = build_cv_results(
            fold_results=fold_results,
            summary=summary,
            config=OmegaConf.to_container(cfg, resolve=True),
            granularity="volume-wise",
            model_type="volume_wise_medicalnet",
            classifier_type=classifier_type,
            backbone=f"medicalnet_{model_cfg.model_depth}",
            volume_info=vol_info,
        )
        classifier_dir = os.path.join(output_dir, classifier_type)
        os.makedirs(classifier_dir, exist_ok=True)
        results_path = os.path.join(classifier_dir, "results.json")
        save_results(results, results_path)
        logger.info(f"\n✓ Results for {classifier_type} saved to: {results_path}")
        all_classifier_results[classifier_type] = results

    logger.info(f"\n{'='*70}")
    logger.info("ALL CLASSIFIERS COMPLETED")
    logger.info(f"{'='*70}\n")
    logger.info(f"✓ All results saved to {output_dir}")


if __name__ == "__main__":
    main()