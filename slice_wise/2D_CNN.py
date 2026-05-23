"""Baseline 2D CNN for slice-wise tumor grade classification."""

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
from torchvision.models import resnet18

# Allow running this file directly by ensuring the package root (Code/) is importable.
if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.aggregate import aggregate_patients
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
from slice_wise.data import build_slice_batch, build_slice_index


logger = logging.getLogger(__name__)

_DEFAULT_EPOCHS = 10
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_LR = 1e-4
_DEFAULT_WEIGHT_DECAY = 1e-4


class Slim2DCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x).squeeze(-1)


def _baseline_hyperparameters(cfg: DictConfig) -> tuple[int, int, float, float]:
	epochs = _DEFAULT_EPOCHS
	batch_size = _DEFAULT_BATCH_SIZE
	lr = _DEFAULT_LR
	weight_decay = _DEFAULT_WEIGHT_DECAY

	if "cnn2d" in cfg.model:
		cnn_cfg = cfg.model.cnn2d
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
	return (batch - mean[None, :, None, None]) / std[None, :, None, None]


def _compute_channel_stats(
	df,
	feature_cols,
	metadata_all: np.ndarray,
	train_indices: np.ndarray,
	n_channels: int,
	target_size: int,
	tumor_only: bool,
	batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
	channel_sum = np.zeros(n_channels, dtype=np.float64)
	channel_sumsq = np.zeros(n_channels, dtype=np.float64)
	pixel_count = 0

	for start in range(0, len(train_indices), batch_size):
		batch_indices = train_indices[start : start + batch_size]
		batch_metadata = metadata_all[batch_indices]
		batch = build_slice_batch(
			df=df,
			feature_cols=feature_cols,
			metadata_batch=batch_metadata,
			n_channels=n_channels,
			target_size=target_size,
			tumor_only=tumor_only,
		).astype(np.float64, copy=False)
		channel_sum += batch.sum(axis=(0, 2, 3))
		channel_sumsq += np.square(batch).sum(axis=(0, 2, 3))
		pixel_count += int(batch.shape[0] * batch.shape[2] * batch.shape[3])

	assert pixel_count > 0, "Training split must contain at least one slice"

	mean = channel_sum / pixel_count
	var = np.maximum(channel_sumsq / pixel_count - mean**2, 1e-12)
	std = np.sqrt(var)
	std[std == 0] = 1.0
	return mean.astype(np.float32), std.astype(np.float32)


def _train_model(
	model: nn.Module,
	df,
	feature_cols,
	metadata_all: np.ndarray,
	labels_all: np.ndarray,
	train_indices: np.ndarray,
	n_channels: int,
	target_size: int,
	tumor_only: bool,
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

	n_pos = int(np.sum(labels_all[train_indices] == 1))
	n_neg = int(np.sum(labels_all[train_indices] == 0))
	assert n_pos > 0 and n_neg > 0, "Training fold must contain both classes"
	pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

	rng = np.random.default_rng(seed)

	for epoch in range(epochs):
		shuffled_indices = rng.permutation(train_indices)
		epoch_loss = 0.0
		n_seen = 0

		for start in range(0, len(shuffled_indices), batch_size):
			batch_indices = shuffled_indices[start : start + batch_size]
			batch_metadata = metadata_all[batch_indices]
			batch_x = build_slice_batch(
				df=df,
				feature_cols=feature_cols,
				metadata_batch=batch_metadata,
				n_channels=n_channels,
				target_size=target_size,
				tumor_only=tumor_only,
			)
			batch_x = _normalize_batch(batch_x, mean, std)
			batch_y = labels_all[batch_indices].astype(np.float32, copy=False)

			inputs = torch.from_numpy(batch_x).to(device)
			targets = torch.from_numpy(batch_y).to(device)

			optimizer.zero_grad()
			logits = model(inputs).view(-1)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()

			epoch_loss += float(loss.item()) * len(batch_indices)
			n_seen += len(batch_indices)

			del inputs, targets, logits, loss, batch_x
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		logger.info(f"  Epoch {epoch + 1}/{epochs} | loss={epoch_loss / max(n_seen, 1):.4f}")


def _predict_slice_split(
	model: nn.Module,
	df,
	feature_cols,
	metadata_all: np.ndarray,
	labels_all: np.ndarray,
	patient_ids_all: np.ndarray,
	indices: np.ndarray,
	n_channels: int,
	target_size: int,
	tumor_only: bool,
	mean: np.ndarray,
	std: np.ndarray,
	device: torch.device,
	batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
	model.eval()
	y_pred_list: list[np.ndarray] = []
	y_prob_list: list[np.ndarray] = []

	with torch.no_grad():
		for start in range(0, len(indices), batch_size):
			batch_indices = indices[start : start + batch_size]
			batch_metadata = metadata_all[batch_indices]
			batch_x = build_slice_batch(
				df=df,
				feature_cols=feature_cols,
				metadata_batch=batch_metadata,
				n_channels=n_channels,
				target_size=target_size,
				tumor_only=tumor_only,
			)
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
	y_true = labels_all[indices]
	patient_ids = patient_ids_all[indices]
	metrics = compute_metrics(y_true, y_pred, y_prob)
	return y_pred, y_prob, patient_ids, y_true, metrics


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
	"""Train and evaluate a from-scratch 2D CNN baseline."""

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

	labels_all, patient_ids_all, slice_metadata_all, feature_names, slice_info = build_slice_index(
		df=df,
		feature_cols=cfg.features[dataset_name].feature_cols,
		planes=cfg.features.planes,
		min_tumor_pixels=cfg.features.min_tumor_pixels,
		tumor_only=cfg.features.tumor_only,
	)

	patient_grades = df["grade"].values.astype(int)
	n_folds = int(cfg.data.n_folds)
	n_input_channels = int(slice_info["n_channels"])
	target_size = int(cfg.features.target_size)
	epochs, batch_size, lr, weight_decay = _baseline_hyperparameters(cfg)

	logger.info(f"Total slices: {slice_info['n_slices_total']}")
	logger.info(f"Total channels: {n_input_channels}")
	logger.info(f"Patients: {slice_info['n_patients']} for {n_folds}-fold CV\n")
	logger.info(
		f"2D CNN baseline | epochs={epochs} | batch_size={batch_size} | lr={lr} | weight_decay={weight_decay}"
	)

	fold_results: list[dict] = []

	for fold_idx, (fold_train_patients, fold_test_patients) in enumerate(
		make_cv_folds(patient_ids_all, patient_grades, n_folds, int(cfg.data.random_state))
	):
		logger.info(f"\n{'=' * 70}")
		logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
		logger.info(f"{'=' * 70}\n")

		train_mask = np.isin(patient_ids_all, fold_train_patients)
		test_mask = np.isin(patient_ids_all, fold_test_patients)
		train_indices = np.where(train_mask)[0]
		test_indices = np.where(test_mask)[0]

		y_train = labels_all[train_mask]
		train_pids = patient_ids_all[train_mask]
		y_test = labels_all[test_mask]
		test_pids = patient_ids_all[test_mask]

		logger.info(
			f"Fold {fold_idx + 1}: {len(train_indices)} train slices from {len(fold_train_patients)} patients"
		)
		logger.info(
			f"Fold {fold_idx + 1}: {len(test_indices)} test slices from {len(fold_test_patients)} patients"
		)
		logger.info(
			f"  Train patients: LGG={np.sum(patient_grades[fold_train_patients] == 0)}, "
			f"HGG={np.sum(patient_grades[fold_train_patients] == 1)}"
		)
		logger.info(
			f"  Test patients: LGG={np.sum(patient_grades[fold_test_patients] == 0)}, "
			f"HGG={np.sum(patient_grades[fold_test_patients] == 1)}"
		)

		mean, std = _compute_channel_stats(
			df=df,
			feature_cols=cfg.features[dataset_name].feature_cols,
			metadata_all=slice_metadata_all,
			train_indices=train_indices,
			n_channels=n_input_channels,
			target_size=target_size,
			tumor_only=cfg.features.tumor_only,
			batch_size=batch_size,
		)

		model = Slim2DCNN(in_channels=n_input_channels).to(device)
		n_trainable_torch_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
		n_total_torch_params = int(sum(p.numel() for p in model.parameters()))

		_train_model(
			model=model,
			df=df,
			feature_cols=cfg.features[dataset_name].feature_cols,
			metadata_all=slice_metadata_all,
			labels_all=labels_all,
			train_indices=train_indices,
			n_channels=n_input_channels,
			target_size=target_size,
			tumor_only=cfg.features.tumor_only,
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

		train_pred, train_prob, train_pids_eval, train_true_eval, train_slice_metrics = _predict_slice_split(
			model=model,
			df=df,
			feature_cols=cfg.features[dataset_name].feature_cols,
			metadata_all=slice_metadata_all,
			labels_all=labels_all,
			patient_ids_all=patient_ids_all,
			indices=train_indices,
			n_channels=n_input_channels,
			target_size=target_size,
			tumor_only=cfg.features.tumor_only,
			mean=mean,
			std=std,
			device=device,
			batch_size=int(cfg.train.inference_batch_size),
		)
		test_pred, test_prob, test_pids_eval, test_true_eval, test_slice_metrics = _predict_slice_split(
			model=model,
			df=df,
			feature_cols=cfg.features[dataset_name].feature_cols,
			metadata_all=slice_metadata_all,
			labels_all=labels_all,
			patient_ids_all=patient_ids_all,
			indices=test_indices,
			n_channels=n_input_channels,
			target_size=target_size,
			tumor_only=cfg.features.tumor_only,
			mean=mean,
			std=std,
			device=device,
			batch_size=int(cfg.train.inference_batch_size),
		)

		(
			train_pred_patient,
			train_label_patient,
			train_prob_patient,
		), (
			test_pred_patient,
			test_label_patient,
			test_prob_patient,
		) = aggregate_patients(
			method=cfg.patient_aggregation,
			train_prob=train_prob,
			train_true=train_true_eval,
			train_pids=train_pids_eval,
			test_prob=test_prob,
			test_true=test_true_eval,
			test_pids=test_pids_eval,
			mil_epochs=cfg.mil.epochs,
			mil_lr=cfg.mil.lr,
		)

		train_patient_metrics = compute_metrics(train_label_patient, train_pred_patient, train_prob_patient)
		test_patient_metrics = compute_metrics(test_label_patient, test_pred_patient, test_prob_patient)

		logger.info(f"\nFold {fold_idx + 1} Results:")
		logger.info(f"  Slice-level train: {compact_metrics_for_log(train_slice_metrics)}")
		logger.info(f"  Slice-level test:  {compact_metrics_for_log(test_slice_metrics)}")
		logger.info(f"  Patient-level train: {compact_metrics_for_log(train_patient_metrics)}")
		logger.info(f"  Patient-level test:  {compact_metrics_for_log(test_patient_metrics)}")

		fold_results.append(
			{
				"fold": fold_idx + 1,
				"n_train": int(len(fold_train_patients)),
				"n_test": int(len(fold_test_patients)),
				"train": train_patient_metrics,
				"test": test_patient_metrics,
				"train_slice": train_slice_metrics,
				"test_slice": test_slice_metrics,
				"history": None,
				"n_trainable_params_estimate": n_trainable_params_estimate,
				"n_trainable_torch_params": n_trainable_torch_params,
				"n_total_torch_params": n_total_torch_params,
			}
		)

		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	summary = summarise_cv_folds(fold_results)
	log_cv_summary(logger, summary)

	results = build_cv_results(
		fold_results=fold_results,
		summary=summary,
		config=OmegaConf.to_container(cfg, resolve=True),
		granularity="slice-wise",
		n_input_channels=n_input_channels,
		feature_names=feature_names,
	)

	baseline_dir = output_dir / "2d_cnn"
	baseline_dir.mkdir(parents=True, exist_ok=True)
	save_results(results, str(baseline_dir / "results.json"))

	logger.info(f"\n✓ 2D CNN baseline saved to {baseline_dir}")


if __name__ == "__main__":
	main()
