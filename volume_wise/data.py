"""Data preparation for volume-wise 3D classification."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def resample_volume(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resample a 3D (D, H, W) or 4D (D, H, W, T) volume to target spatial shape.

    For 4D volumes only the spatial dims are resampled; T is unchanged.
    Uses linear (order=1) interpolation.
    """
    spatial = vol.shape[:3]
    zoom_factors = tuple(t / s for t, s in zip(target_shape, spatial))
    if vol.ndim == 3:
        return zoom(vol, zoom_factors, order=1).astype(np.float32)
    elif vol.ndim == 4:
        return zoom(vol, (*zoom_factors, 1.0), order=1).astype(np.float32)
    else:
        raise ValueError(f"Volume must be 3D or 4D, got shape {vol.shape}")


def extract_patient_volumes(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_shape: Tuple[int, int, int],
    tumor_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """Build per-patient 3D volume tensors stacked across modalities.

    4D modalities (e.g. PET_dynamic with T timeframes) are expanded into T
    separate channels, mirroring the slice-wise convention.

    Parameters
    ----------
    df : DataFrame with columns 'GT', 'grade', and one column per modality.
    feature_cols : modality names to stack as channels.
    target_shape : (D, H, W) — all volumes resampled to this shape.
    tumor_only : if True, non-tumour voxels are zeroed out.

    Returns
    -------
    X_all       : (N, C, D, H, W)  float32
    y_all       : (N,)  int64
    patient_ids : (N,)  int64  (0-based index into df)
    feature_names : list of channel names (expanded for 4D modalities)
    info        : summary dict
    """
    logger.info(f"Extracting 3D volumes — target_shape={target_shape}, tumor_only={tumor_only}")

    # Expand feature names for 4D volumes
    expanded_features: List[str] = []
    for col in feature_cols:
        sample = df[col].iloc[0]
        if sample.ndim == 4:
            n_t = sample.shape[-1]
            expanded_features.extend([f"{col}_t{t}" for t in range(n_t)])
        else:
            expanded_features.append(col)

    n_channels = len(expanded_features)
    n_patients = len(df)
    D, H, W = target_shape

    memory_gb = (n_patients * n_channels * D * H * W * 4) / 1024 ** 3
    logger.info(f"Channels: {n_channels}  |  Patients: {n_patients}  |  "
                f"Shape per volume: ({D},{H},{W})  |  Estimated memory: {memory_gb:.2f} GB")

    X_all = np.zeros((n_patients, n_channels, D, H, W), dtype=np.float32)
    y_all = np.zeros(n_patients, dtype=np.int64)
    patient_ids = np.arange(n_patients, dtype=np.int64)
    nan_total = 0

    for pidx in range(n_patients):
        # Ground-truth mask
        GT = df['GT'].iloc[pidx]
        if GT.ndim == 4:
            GT = GT[:, :, :, 0]
        GT_resampled = resample_volume(GT, target_shape)
        tumor_mask = (GT_resampled > 0.5).astype(np.float32)  # (D, H, W)

        # Label
        label = int(df['grade'].iloc[pidx])

        # Fill channels
        ch_idx = 0
        for col in feature_cols:
            vol = df[col].iloc[pidx]
            resampled = resample_volume(vol, target_shape)  # (D,H,W) or (D,H,W,T)
            if resampled.ndim == 3:
                if tumor_only:
                    resampled = resampled * tumor_mask
                X_all[pidx, ch_idx] = resampled
                ch_idx += 1
            elif resampled.ndim == 4:
                for t in range(resampled.shape[-1]):
                    frame = resampled[:, :, :, t]
                    if tumor_only:
                        frame = frame * tumor_mask
                    X_all[pidx, ch_idx] = frame
                    ch_idx += 1

        # Handle NaN
        n_nan = int(np.sum(np.isnan(X_all[pidx])))
        if n_nan > 0:
            X_all[pidx] = np.nan_to_num(X_all[pidx], nan=0.0)
            nan_total += n_nan

        y_all[pidx] = label

        if (pidx + 1) % 10 == 0 or pidx == n_patients - 1:
            logger.info(f"  Patient {pidx + 1}/{n_patients} done")

    info = {
        'n_patients': n_patients,
        'n_channels': n_channels,
        'target_shape': target_shape,
        'feature_names': expanded_features,
        'tumor_only': tumor_only,
        'n_nan_replaced': nan_total,
        'label_distribution': {
            'low_grade': int(np.sum(y_all == 0)),
            'high_grade': int(np.sum(y_all == 1)),
        },
    }

    logger.info(f"Extraction complete — X_all: {X_all.shape}, "
                f"Low={info['label_distribution']['low_grade']}, "
                f"High={info['label_distribution']['high_grade']}")
    return X_all, y_all, patient_ids, expanded_features, info


def scale_fold_split(
    X_all: np.ndarray,
    y_all: np.ndarray,
    patient_ids: np.ndarray,
    fold_train_patients: np.ndarray,
    fold_test_patients: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Return normalised train/test arrays for a CV fold.

    Reshapes (N, C, D, H, W) → (N*D*H*W, C) so StandardScaler normalises
    per channel (one mean/std per modality), then restores the original shape.
    Scaler is fit on training data only to prevent data leakage.

    Returns
    -------
    X_train, y_train, X_test, y_test, scaler
    """
    train_mask = np.isin(patient_ids, fold_train_patients)
    test_mask = np.isin(patient_ids, fold_test_patients)

    X_train = X_all[train_mask].copy()
    y_train = y_all[train_mask]
    X_test = X_all[test_mask].copy()
    y_test = y_all[test_mask]

    N_tr, C, D, H, W = X_train.shape
    N_te = X_test.shape[0]

    # Calculate means and stds across all axes except channels (axis 1)
    means = np.mean(X_train, axis=(0, 2, 3, 4), keepdims=True)
    stds = np.std(X_train, axis=(0, 2, 3, 4), keepdims=True)
    
    # Avoid division by zero
    stds[stds == 0] = 1.0

    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds

    # Create dummy scaler for compatibility
    scaler = StandardScaler()
    scaler.mean_ = means.flatten()
    scaler.scale_ = stds.flatten()
    scaler.var_ = stds.flatten() ** 2

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler