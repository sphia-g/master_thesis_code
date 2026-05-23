"""Baseline 3D CNN for volume-wise tumor grade classification."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchvision.models.video import r3d_18


# Allow running this file directly by ensuring the package root (Code/) is importable.
if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.data import make_cv_folds
from shared.data_loaders import get_loader
from shared.evaluate import (
    build_cv_results,
    compact_metrics_for_log,
    compute_metrics,
    estimate_trainable_parameters,
    log_cv_summary,
    summarise_cv_folds,
)
from shared.utils import save_results, set_determinism
from volume_wise.data import extract_patient_volumes, scale_fold_split


logger = logging.getLogger(__name__)

_DEFAULT_EPOCHS = 10
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_LR = 1e-4
_DEFAULT_WEIGHT_DECAY = 1e-4


class Slim3DCNN(nn.Module):
    """Small 3D CNN trained from scratch on volume tensors."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        backbone = r3d_18(weights=None)
        backbone.stem[0] = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        nn.init.kaiming_normal_(backbone.stem[0].weight, mode="fan_out", nonlinearity="relu")
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def _baseline_hyperparameters(cfg: DictConfig) -> tuple[int, int, float, float]:
    epochs = _DEFAULT_EPOCHS
    batch_size = _DEFAULT_BATCH_SIZE
    lr = _DEFAULT_LR
    weight_decay = _DEFAULT_WEIGHT_DECAY

    if "cnn3d" in cfg.model:
        cnn_cfg = cfg.model.cnn3d
        if "epochs" in cnn_cfg:
            epochs = int(cnn_cfg.epochs)
        if "batch_size" in cnn_cfg:
            batch_size = int(cnn_cfg.batch_size)
        if "lr" in cnn_cfg:
            lr = float(cnn_cfg.lr)
        if "weight_decay" in cnn_cfg:
            weight_decay = float(cnn_cfg.weight_decay)

    return epochs, batch_size, lr, weight_decay


def _normalize_batch(batch: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (batch - mean[None, :, None, None, None]) / std[None, :, None, None, None]


def _compute_channel_stats(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from training volumes."""
    channel_mean = np.mean(X_train, axis=(0, 2, 3, 4))
    channel_std = np.std(X_train, axis=(0, 2, 3, 4))
    channel_std[channel_std == 0] = 1.0
    return channel_mean.astype(np.float32), channel_std.astype(np.float32)


def _train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    assert n_pos > 0 and n_neg > 0, "Training fold must contain both classes"
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        shuffled_idx = rng.permutation(len(X_train))
        epoch_loss = 0.0
        n_seen = 0

        for start in range(0, len(shuffled_idx), batch_size):
            batch_idx = shuffled_idx[start : start + batch_size]
            batch_x = X_train[batch_idx]
            batch_x = _normalize_batch(batch_x, mean, std)
            batch_y = y_train[batch_idx].astype(np.float32, copy=False)

            inputs = torch.from_numpy(batch_x).to(device)
            targets = torch.from_numpy(batch_y).to(device)

            optimizer.zero_grad()
            logits = model(inputs).view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(batch_idx)
            n_seen += len(batch_idx)

            del inputs, targets, logits, loss, batch_x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"  Epoch {epoch + 1}/{epochs} | loss={epoch_loss / max(n_seen, 1):.4f}")


def _predict_split(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()
    y_pred_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch_x = X[start : start + batch_size]
            batch_x = _normalize_batch(batch_x, mean, std)
            inputs = torch.from_numpy(batch_x).to(device)
            logits = model(inputs).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int64)

            y_pred_list.append(preds.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

            del inputs, logits, probs, preds, batch_x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list)
    metrics = compute_metrics(y, y_pred, y_prob)
    return y_pred, y_prob, metrics


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train and evaluate a from-scratch 3D CNN baseline."""

    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Output directory: {output_dir}")

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "Data"

    set_determinism(int(cfg.train.seed))
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_name = cfg.dataset.name
    dataset_config = cfg.dataset[dataset_name]
    logger.info(f"Loading {dataset_name} dataset...")
    df = get_loader(dataset_name, dataset_config, data_dir).load()

    target_shape = tuple(cfg.features.target_shape)
    X_all, y_all, patient_ids, feature_names, vol_info = extract_patient_volumes(
        df=df,
        feature_cols=cfg.features[dataset_name].feature_cols,
        target_shape=target_shape,
        tumor_only=cfg.features.tumor_only,
    )

    patient_grades = df["grade"].values.astype(int)
    n_folds = int(cfg.data.n_folds)
    n_input_channels = int(X_all.shape[1])
    epochs, batch_size, lr, weight_decay = _baseline_hyperparameters(cfg)

    logger.info(f"Total patients: {len(X_all)}")
    logger.info(f"Total channels: {n_input_channels}")
    logger.info(f"Patients: {len(X_all)} for {n_folds}-fold CV\n")
    logger.info(
        f"3D CNN baseline | epochs={epochs} | batch_size={batch_size} | lr={lr} | weight_decay={weight_decay}"
    )

    fold_results: list[dict] = []

    for fold_idx, (fold_train_patients, fold_test_patients) in enumerate(
        make_cv_folds(patient_ids, patient_grades, n_folds, int(cfg.data.random_state))
    ):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
        logger.info(f"{'=' * 70}\n")

        train_mask = np.isin(patient_ids, fold_train_patients)
        test_mask = np.isin(patient_ids, fold_test_patients)

        X_train_raw = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test_raw = X_all[test_mask]
        y_test = y_all[test_mask]

        logger.info(f"Fold {fold_idx + 1}: {len(X_train_raw)} train patients")
        logger.info(f"Fold {fold_idx + 1}: {len(X_test_raw)} test patients")
        logger.info(
            f"  Train patients: LGG={np.sum(patient_grades[fold_train_patients] == 0)}, "
            f"HGG={np.sum(patient_grades[fold_train_patients] == 1)}"
        )
        logger.info(
            f"  Test patients: LGG={np.sum(patient_grades[fold_test_patients] == 0)}, "
            f"HGG={np.sum(patient_grades[fold_test_patients] == 1)}"
        )

        mean, std = _compute_channel_stats(X_train_raw)

        model = Slim3DCNN(in_channels=n_input_channels).to(device)
        n_trainable_torch_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        n_total_torch_params = int(sum(p.numel() for p in model.parameters()))

        _train_model(
            model=model,
            X_train=X_train_raw,
            y_train=y_train,
            mean=mean,
            std=std,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=int(cfg.train.seed) + fold_idx,
        )

        n_trainable_params_estimate = estimate_trainable_parameters(model)

        train_pred, train_prob, train_metrics = _predict_split(
            model=model,
            X=X_train_raw,
            y=y_train,
            mean=mean,
            std=std,
            device=device,
            batch_size=int(cfg.train.inference_batch_size),
        )
        test_pred, test_prob, test_metrics = _predict_split(
            model=model,
            X=X_test_raw,
            y=y_test,
            mean=mean,
            std=std,
            device=device,
            batch_size=int(cfg.train.inference_batch_size),
        )

        logger.info(f"\nFold {fold_idx + 1} Results (patient-level):")
        logger.info(f"  Train: {compact_metrics_for_log(train_metrics)}")
        logger.info(f"  Test:  {compact_metrics_for_log(test_metrics)}")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "n_train": int(len(X_train_raw)),
                "n_test": int(len(X_test_raw)),
                "train": train_metrics,
                "test": test_metrics,
                "history": None,
                "n_trainable_params_estimate": n_trainable_params_estimate,
                "n_trainable_torch_params": n_trainable_torch_params,
                "n_total_torch_params": n_total_torch_params,
            }
        )

        del model, X_train_raw, X_test_raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarise_cv_folds(fold_results)
    log_cv_summary(logger, summary)

    results = build_cv_results(
        fold_results=fold_results,
        summary=summary,
        config=OmegaConf.to_container(cfg, resolve=True),
        granularity="volume-wise",
        n_input_channels=n_input_channels,
        feature_names=feature_names,
        volume_info=vol_info,
    )

    baseline_dir = output_dir / "3d_cnn"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, str(baseline_dir / "results.json"))

    logger.info(f"\n✓ 3D CNN baseline saved to {baseline_dir}")


if __name__ == "__main__":
    main()
