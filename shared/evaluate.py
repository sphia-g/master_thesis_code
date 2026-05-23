"""Evaluation utilities for classification models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)

# Scalar metric keys that summarise_cv_folds aggregates across folds
_SCALAR_METRIC_KEYS = (
    "accuracy", "balanced_accuracy", "mcc",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "roc_auc",
)

# Decision thresholds for additional confusion-matrix based diagnostics.
_THRESHOLD_GRID = tuple(np.round(np.linspace(0.1, 0.9, 9), 2))
_FOLD_PARAM_KEYS = (
    'n_trainable_params_estimate',
    'n_trainable_torch_params',
    'n_total_torch_params',
)


def _count_trainable_torch_parameters(model: Any) -> int:
    """Return trainable torch parameter count for nn.Module objects."""
    if not isinstance(model, nn.Module):
        return 0
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _estimate_fitted_sklearn_parameters(model: Any) -> int:
    """Estimate fitted sklearn parameters from learned attributes ending in '_' ."""
    estimator = model._model if hasattr(model, '_model') else model
    if not hasattr(estimator, '__dict__'):
        return 0

    total = 0
    for name, value in estimator.__dict__.items():
        if not name.endswith('_') or name.startswith('__'):
            continue
        if isinstance(value, np.ndarray):
            total += int(value.size)
        elif np.isscalar(value):
            total += 1
    return int(total)


def estimate_trainable_parameters(model: Any) -> int:
    """Estimate trainable parameter count as one number for reporting.

    Includes:
    - trainable torch parameters
    - fitted sklearn-style parameters for classifier wrappers
    """
    total = _count_trainable_torch_parameters(model)

    if hasattr(model, 'classifier'):
        total += _estimate_fitted_sklearn_parameters(model.classifier)
    else:
        total += _estimate_fitted_sklearn_parameters(model)

    if hasattr(model, 'base_classifier'):
        total += _estimate_fitted_sklearn_parameters(model.base_classifier)

    return int(total)


def compact_metrics_for_log(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Return only compact scalar metrics for concise logging."""
    keys = ('accuracy', 'balanced_accuracy', 'mcc', 'f1_macro', 'roc_auc')
    return {k: float(metrics[k]) for k in keys if k in metrics}


def _safe_ratio(num: float, den: float) -> float:
    """Return num/den with 0.0 when den is zero."""
    return float(num / den) if den > 0 else 0.0


def _compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, Any]]:
    """Compute confusion-matrix based metrics for a fixed threshold grid."""
    threshold_metrics: List[Dict[str, Any]] = []

    for thr in _THRESHOLD_GRID:
        y_pred_thr = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred_thr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = _safe_ratio(tp, tp + fn)
        specificity = _safe_ratio(tn, tn + fp)
        precision_pos = _safe_ratio(tp, tp + fp)

        threshold_metrics.append(
            {
                "threshold": float(thr),
                "confusion_matrix": cm,
                "accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
                "balanced_accuracy": 0.5 * (sensitivity + specificity),
                "precision_positive": precision_pos,
                "recall_positive": sensitivity,
                "specificity": specificity,
                "fpr": _safe_ratio(fp, fp + tn),
                "tpr": sensitivity,
            }
        )

    return threshold_metrics


def _summarise_threshold_metrics_across_folds(fold_results: List[dict], split: str) -> List[Dict[str, Any]]:
    """Aggregate threshold metrics across folds as mean ± std."""
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "precision_positive",
        "recall_positive",
        "specificity",
        "fpr",
        "tpr",
    )
    first_split_metrics = fold_results[0][split]["threshold_metrics"]
    summary: List[Dict[str, Any]] = []

    for idx, threshold_entry in enumerate(first_split_metrics):
        thr = float(threshold_entry["threshold"])
        entry_summary: Dict[str, Any] = {"threshold": thr}

        cms = np.asarray(
            [f[split]["threshold_metrics"][idx]["confusion_matrix"] for f in fold_results],
            dtype=float,
        )
        entry_summary["confusion_matrix"] = {
            "mean": np.mean(cms, axis=0),
            "std": np.std(cms, axis=0),
        }

        for key in metric_keys:
            vals = [float(f[split]["threshold_metrics"][idx][key]) for f in fold_results]
            entry_summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        summary.append(entry_summary)

    return summary

def get_predictions_batch(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    apply_sigmoid: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions in batches.

    Returns (predicted_labels, predicted_probabilities).
    """
    model.eval()
    preds_list, probs_list = [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = torch.FloatTensor(X[i : i + batch_size]).to(device)
            batch_outputs = model(batch_X).squeeze(-1)
            batch_probs = torch.sigmoid(batch_outputs) if apply_sigmoid else batch_outputs
            batch_preds = (batch_probs >= 0.5).float()
            preds_list.append(batch_preds.cpu().numpy())
            probs_list.append(batch_probs.cpu().numpy())
            del batch_X, batch_outputs, batch_probs, batch_preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.concatenate(preds_list), np.concatenate(probs_list)


def evaluate_estimator(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device | None = None,
    batch_size: int = 32,
    apply_sigmoid: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluate any model; returns (predictions, probabilities, metrics).

    Routes to sklearn predict / predict_proba when the model exposes that API
    (e.g. TabPFN). Falls back to batched torch forward pass otherwise (requires
    device to be provided).
    """
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    else:
        assert device is not None, "device must be provided for nn.Module models without predict_proba"
        y_pred, y_prob = get_predictions_batch(model, X, device, batch_size, apply_sigmoid)
    return y_pred, y_prob, compute_metrics(y, y_pred, y_prob)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive binary classification metrics.

    Parameters
    ----------
    y_true : array of 0/1 true labels
    y_pred : array of 0/1 predicted labels
    y_prob : predicted probabilities for the positive class

    Returns a dict with scalar metrics (accuracy, balanced_accuracy, mcc,
    precision/recall/f1 per-class and macro/weighted averages, roc_auc) plus
    confusion_matrix and support arrays.
    """
    assert len(np.unique(y_true)) == 2, (
        f"Binary classification requires both classes in y_true, got {np.unique(y_true)}"
    )
    assert len(y_true) == len(y_pred) == len(y_prob), (
        "y_true, y_pred, y_prob must have the same length"
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, y_prob)
    finite_thresholds = roc_thresholds[np.isfinite(roc_thresholds)]

    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "mcc":                matthews_corrcoef(y_true, y_pred),
        "confusion_matrix":   confusion_matrix(y_true, y_pred),
        # Per-class arrays [LGG, HGG]
        "precision":          precision,
        "recall":             recall,
        "f1":                 f1,
        "support":            support,
        # Macro (equal weight per class — preferred for imbalanced data)
        "precision_macro":    precision_macro,
        "recall_macro":       recall_macro,
        "f1_macro":           f1_macro,
        # Weighted by class frequency
        "precision_weighted": precision_weighted,
        "recall_weighted":    recall_weighted,
        "f1_weighted":        f1_weighted,
        "roc_auc":            roc_auc_score(y_true, y_prob),
        # Full ROC curve points across score thresholds.
        "roc_curve": {
            "fpr": roc_fpr,
            "tpr": roc_tpr,
            "thresholds": finite_thresholds,
        },
        # Confusion-matrix diagnostics at fixed decision thresholds.
        "threshold_metrics": _compute_threshold_metrics(y_true, y_prob),
    }


def summarise_cv_folds(fold_results: List[dict]) -> dict:
    """Compute mean ± std for all scalar metrics across CV folds.

    Each fold_result must have 'train' and 'test' keys mapping to dicts from
    compute_metrics. Optional 'train_slice' / 'test_slice' are also summarised
    when present (slice-wise experiments).
    """
    assert len(fold_results) > 0, "fold_results must not be empty"

    splits = ["train", "test"]
    if "train_slice" in fold_results[0]:
        splits += ["train_slice", "test_slice"]

    summary: Dict[str, Any] = {"n_folds": len(fold_results)}
    for split in splits:
        split_summary = {}
        for key in _SCALAR_METRIC_KEYS:
            vals = [float(f[split][key]) for f in fold_results]
            split_summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        if "threshold_metrics" in fold_results[0][split]:
            split_summary["threshold_metrics"] = _summarise_threshold_metrics_across_folds(
                fold_results, split
            )
        summary[split] = split_summary

    overfit_gaps = [f["train"]["accuracy"] - f["test"]["accuracy"] for f in fold_results]
    summary["avg_overfit_gap"] = float(np.mean(overfit_gaps))

    test_roc_aucs = [f["test"]["roc_auc"] for f in fold_results]
    summary["best_fold"] = int(fold_results[int(np.argmax(test_roc_aucs))]["fold"])

    for key in _FOLD_PARAM_KEYS:
        if all(key in fold for fold in fold_results):
            vals = [int(fold[key]) for fold in fold_results]
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return summary


def log_cv_summary(logger: logging.Logger, summary: dict) -> None:
    """Log a cross-validation summary produced by summarise_cv_folds."""
    n = summary["n_folds"]
    tr, te = summary["train"], summary["test"]
    logger.info(f"\nCross-Validation Summary ({n} folds):")
    logger.info(
        f"  Train: Acc={tr['accuracy']['mean']:.4f}±{tr['accuracy']['std']:.4f}  "
        f"BalAcc={tr['balanced_accuracy']['mean']:.4f}±{tr['balanced_accuracy']['std']:.4f}  "
        f"MCC={tr['mcc']['mean']:.4f}±{tr['mcc']['std']:.4f}  "
        f"F1={tr['f1_macro']['mean']:.4f}±{tr['f1_macro']['std']:.4f}  "
        f"AUC={tr['roc_auc']['mean']:.4f}±{tr['roc_auc']['std']:.4f}"
    )
    logger.info(
        f"  Test:  Acc={te['accuracy']['mean']:.4f}±{te['accuracy']['std']:.4f}  "
        f"BalAcc={te['balanced_accuracy']['mean']:.4f}±{te['balanced_accuracy']['std']:.4f}  "
        f"MCC={te['mcc']['mean']:.4f}±{te['mcc']['std']:.4f}  "
        f"F1={te['f1_macro']['mean']:.4f}±{te['f1_macro']['std']:.4f}  "
        f"AUC={te['roc_auc']['mean']:.4f}±{te['roc_auc']['std']:.4f}"
    )
    logger.info(f"  Avg overfitting gap: {summary['avg_overfit_gap']:+.4f}")
    logger.info(f"  Best fold: {summary['best_fold']} (by test ROC-AUC)")


def build_cv_results(
    fold_results: List[dict],
    summary: dict,
    config: dict,
    granularity: str,
    **metadata: Any,
) -> dict:
    """Build a standardised results dict for saving to JSON via save_results.

    Parameters
    ----------
    fold_results : per-fold dicts (each with 'train'/'test' sub-dicts)
    summary      : output of summarise_cv_folds
    config       : resolved config as plain dict (OmegaConf.to_container)
    granularity  : 'voxel-wise' | 'slice-wise' | 'volume-wise'
    **metadata   : additional top-level entries (feature_info, model_type, …)
    """
    return {
        "granularity": granularity,
        "cross_validation": {
            "n_folds":      summary["n_folds"],
            "fold_results": fold_results,
            "summary":      summary,
        },
        "config": config,
        **metadata,
    }