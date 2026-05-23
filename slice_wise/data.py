"""Data preparation for slice-wise classification."""

from __future__ import annotations

import logging
from typing import List, Literal, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Axis index for each plane (used in both first and second pass)
_PLANE_AXIS = {'axial': 2, 'coronal': 1, 'sagittal': 0}


def _extract_slice(
    volume: np.ndarray,
    slice_idx: int,
    plane: Literal['axial', 'coronal', 'sagittal'],
) -> np.ndarray:
    """Extract a 2D slice from a 3D volume, or a 2D+T slice from a 4D volume.

    Uses np.take along the plane's spatial axis, so it works identically for
    3D (returns shape H×W) and 4D (returns shape H×W×T) volumes.
    """
    assert volume.ndim in (3, 4), f"Volume must be 3D or 4D, got {volume.shape}"
    return np.take(volume, slice_idx, axis=_PLANE_AXIS[plane])


def _get_tumor_slice_indices(
    GT: np.ndarray,
    plane: Literal['axial', 'coronal', 'sagittal'],
    min_tumor_pixels: int,
) -> list[int]:
    """Return indices of slices containing at least `min_tumor_pixels` tumor voxels."""
    assert GT.ndim == 3, f"GT must be 3D, got {GT.shape}"
    return [
        i for i in range(GT.shape[_PLANE_AXIS[plane]])
        if np.sum(_extract_slice(GT, i, plane) == 1) >= min_tumor_pixels
    ]


def pad_and_resize_slice(
    slice_image: np.ndarray,
    target_size: int = 224,
    padding_value: float = 0.0,
) -> np.ndarray:
    """Pad a multi-channel 2D slice to a square and resize to target_size.

    Parameters
    ----------
    slice_image : np.ndarray, shape (C, H, W)
    target_size : int
    padding_value : float

    Returns
    -------
    np.ndarray, shape (C, target_size, target_size)
    """
    _, h, w = slice_image.shape
    max_dim = max(h, w)
    pad_h, pad_w = (max_dim - h) // 2, (max_dim - w) // 2
    padded = np.pad(
        slice_image,
        ((0, 0), (pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
        mode='constant',
        constant_values=padding_value,
    )
    if max_dim == target_size:
        return padded
    zoom_factor = target_size / max_dim
    return zoom(padded, (1, zoom_factor, zoom_factor), order=1)


def build_slice_index(
    df: pd.DataFrame,
    feature_cols: List[str],
    planes: List[Literal['axial', 'coronal', 'sagittal']],
    min_tumor_pixels: int = 50,
    tumor_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """Build a compact per-slice index without materializing image tensors.

    Returns label/patient arrays and metadata tuples (patient_idx, plane, slice_idx)
    that can be turned into tensors lazily in batches.
    """
    feature_names: List[str] = []
    for col in feature_cols:
        vol = df[col].iloc[0]
        if vol.ndim == 4:
            feature_names.extend(f"{col}_t{t}" for t in range(vol.shape[-1]))
        else:
            feature_names.append(col)

    mode_str = f"tumor only (min {min_tumor_pixels} px)" if tumor_only else "all slices"
    logger.info(f"Indexing slices | planes={planes} | mode={mode_str} | channels={len(feature_names)}")

    labels: list[int] = []
    patient_ids: list[int] = []
    metadata: list[tuple[int, str, int]] = []
    slices_per_plane = {plane: 0 for plane in planes}

    for patient_idx in range(len(df)):
        GT = df['GT'].iloc[patient_idx]
        if GT.ndim == 4:
            GT = GT[:, :, :, 0]

        label = int(df['grade'].iloc[patient_idx])
        assert label in (0, 1), f"Patient {patient_idx}: unexpected grade {label!r}"

        for plane in planes:
            valid_indices = (
                _get_tumor_slice_indices(GT, plane, min_tumor_pixels)
                if tumor_only
                else list(range(GT.shape[_PLANE_AXIS[plane]]))
            )
            for slice_idx in valid_indices:
                labels.append(label)
                patient_ids.append(patient_idx)
                metadata.append((patient_idx, plane, int(slice_idx)))
                slices_per_plane[plane] += 1

        if (patient_idx + 1) % 10 == 0 or patient_idx == len(df) - 1:
            logger.info(f"  Indexed patient {patient_idx + 1}/{len(df)} | slices so far: {len(labels)}")

    labels_array = np.asarray(labels, dtype=np.int64)
    patient_ids_array = np.asarray(patient_ids, dtype=np.int64)
    metadata_array = np.asarray(metadata, dtype=object)

    info = {
        'n_slices_total': int(len(labels_array)),
        'n_channels': int(len(feature_names)),
        'n_patients': int(len(df)),
        'feature_names': feature_names,
        'planes': list(planes),
        'slices_per_plane': slices_per_plane,
        'min_tumor_pixels': int(min_tumor_pixels),
        'label_distribution': {
            'low_grade': int(np.sum(labels_array == 0)),
            'high_grade': int(np.sum(labels_array == 1)),
        },
    }
    logger.info(
        f"Index complete | n_slices={info['n_slices_total']:,} | "
        f"slices/plane={slices_per_plane} | "
        f"labels: Low={info['label_distribution']['low_grade']}, High={info['label_distribution']['high_grade']}"
    )
    return labels_array, patient_ids_array, metadata_array, feature_names, info


def build_slice_batch(
    df: pd.DataFrame,
    feature_cols: List[str],
    metadata_batch: np.ndarray,
    n_channels: int,
    target_size: int = 224,
    tumor_only: bool = True,
) -> np.ndarray:
    """Materialize a batch of indexed slices as (B, C, H, W) float32 tensors."""
    B = len(metadata_batch)
    X_batch = np.zeros((B, n_channels, target_size, target_size), dtype=np.float32)

    for i, (patient_idx, plane, slice_idx) in enumerate(metadata_batch):
        patient_idx = int(patient_idx)
        slice_idx = int(slice_idx)

        GT = df['GT'].iloc[patient_idx]
        if GT.ndim == 4:
            GT = GT[:, :, :, 0]
        tumor_mask = _extract_slice(GT, slice_idx, plane)

        slice_image = np.zeros((n_channels, *tumor_mask.shape), dtype=np.float32)
        feat_idx = 0
        for col in feature_cols:
            vol = df[col].iloc[patient_idx]
            raw = _extract_slice(vol, slice_idx, plane)
            if raw.ndim == 2:
                slice_image[feat_idx] = raw
                feat_idx += 1
            else:
                for t in range(raw.shape[-1]):
                    slice_image[feat_idx] = raw[:, :, t]
                    feat_idx += 1

        if tumor_only:
            slice_image[:, tumor_mask != 1] = 0.0

        np.nan_to_num(slice_image, nan=0.0, copy=False)
        X_batch[i] = pad_and_resize_slice(slice_image, target_size)

    return X_batch

def extract_slice_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    planes: List[Literal['axial', 'coronal', 'sagittal']],
    min_tumor_pixels: int = 50,
    tumor_only: bool = True,
    target_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """Extract 2D slice images from patient volumes for CNN input.

    Two-pass approach: first count total slices for pre-allocation, then fill.

    Parameters
    ----------
    df : pd.DataFrame
        Patient dataframe with imaging modalities, 'GT' mask, and 'grade' column.
    feature_cols : list[str]
        Modality columns to use. 4D volumes are expanded into one channel per timeframe.
    planes : list[str]
        Anatomical planes to extract slices from.
    min_tumor_pixels : int
        Minimum tumor pixels per slice (only applies when tumor_only=True).
    tumor_only : bool
        If True, extract only slices containing tumor and mask non-tumor pixels to 0.
        If False, extract all slices.
    target_size : int
        Output spatial size — slices are padded to square then resized to (target_size, target_size).

    Returns
    -------
    slices_array : np.ndarray, shape (N, C, target_size, target_size)
    labels_array : np.ndarray, shape (N,)
    patient_ids_array : np.ndarray, shape (N,)
    metadata_array : np.ndarray, shape (N, 3)  — (patient_idx, plane, slice_idx)
    feature_names : list[str]
    info : dict
    """
    # Expand feature names: 4D volumes become <col>_t0, <col>_t1, ...
    feature_names: List[str] = []
    for col in feature_cols:
        vol = df[col].iloc[0]
        if vol.ndim == 4:
            feature_names.extend(f"{col}_t{t}" for t in range(vol.shape[-1]))
        else:
            feature_names.append(col)
    C = len(feature_names)

    mode_str = f"tumor only (min {min_tumor_pixels} px)" if tumor_only else "all slices"
    logger.info(f"Extracting slices | planes={planes} | size={target_size} | mode={mode_str} | channels={C}")

    # --- FIRST PASS: count total slices for pre-allocation ---
    total_slices = 0
    for patient_idx in range(len(df)):
        GT = df['GT'].iloc[patient_idx]
        if GT.ndim == 4:
            GT = GT[:, :, :, 0]
        for plane in planes:
            if tumor_only:
                total_slices += len(_get_tumor_slice_indices(GT, plane, min_tumor_pixels))
            else:
                total_slices += GT.shape[_PLANE_AXIS[plane]]

    mem_gb = (total_slices * C * target_size * target_size * 4) / (1024 ** 3)
    logger.info(f"Pre-allocating {total_slices:,} slices | estimated memory: {mem_gb:.2f} GB")
    if mem_gb > 50:
        logger.warning(f"Large memory requirement ({mem_gb:.1f} GB) — consider reducing planes or using tumor_only=True")

    slices_array = np.zeros((total_slices, C, target_size, target_size), dtype=np.float32)
    labels_array = np.zeros(total_slices, dtype=np.int64)
    patient_ids_array = np.zeros(total_slices, dtype=np.int64)
    metadata_array = np.empty((total_slices, 3), dtype=object)

    # --- SECOND PASS: extract and store ---
    idx = 0
    total_nan = 0
    slices_per_plane = {plane: 0 for plane in planes}

    for patient_idx in range(len(df)):
        GT = df['GT'].iloc[patient_idx]
        if GT.ndim == 4:
            GT = GT[:, :, :, 0]

        label = int(df['grade'].iloc[patient_idx])
        assert label in (0, 1), f"Patient {patient_idx}: unexpected grade {label!r}"

        for plane in planes:
            valid_indices = (
                _get_tumor_slice_indices(GT, plane, min_tumor_pixels)
                if tumor_only
                else list(range(GT.shape[_PLANE_AXIS[plane]]))
            )

            for slice_idx in valid_indices:
                tumor_mask = _extract_slice(GT, slice_idx, plane)
                slice_image = np.zeros((C, *tumor_mask.shape), dtype=np.float32)

                feat_idx = 0
                for col in feature_cols:
                    vol = df[col].iloc[patient_idx]
                    raw = _extract_slice(vol, slice_idx, plane)
                    if raw.ndim == 2:
                        slice_image[feat_idx] = raw
                        feat_idx += 1
                    else:  # 4D volume: raw.shape == (H, W, T)
                        for t in range(raw.shape[-1]):
                            slice_image[feat_idx] = raw[:, :, t]
                            feat_idx += 1

                if tumor_only:
                    slice_image[:, tumor_mask != 1] = 0.0

                n_nan = int(np.sum(np.isnan(slice_image)))
                if n_nan > 0:
                    np.nan_to_num(slice_image, nan=0.0, copy=False)
                    total_nan += n_nan

                slices_array[idx] = pad_and_resize_slice(slice_image, target_size)
                labels_array[idx] = label
                patient_ids_array[idx] = patient_idx
                metadata_array[idx] = (patient_idx, plane, slice_idx)
                idx += 1
                slices_per_plane[plane] += 1

        if (patient_idx + 1) % 10 == 0 or patient_idx == len(df) - 1:
            logger.info(f"  Patient {patient_idx + 1}/{len(df)} | slices so far: {idx}")

    # Trim pre-allocated arrays if any slices were skipped
    if idx < total_slices:
        logger.info(f"Trimming arrays: {total_slices} → {idx}")
        slices_array = slices_array[:idx]
        labels_array = labels_array[:idx]
        patient_ids_array = patient_ids_array[:idx]
        metadata_array = metadata_array[:idx]

    info = {
        'n_slices_total': idx,
        'n_channels': C,
        'n_patients': len(df),
        'feature_names': feature_names,
        'planes': list(planes),
        'slices_per_plane': slices_per_plane,
        'min_tumor_pixels': min_tumor_pixels,
        'target_size': target_size,
        'n_nan_replaced': total_nan,
        'label_distribution': {
            'low_grade': int(np.sum(labels_array == 0)),
            'high_grade': int(np.sum(labels_array == 1)),
        },
    }
    logger.info(
        f"Extraction complete | shape={slices_array.shape} | "
        f"memory={slices_array.nbytes / (1024**3):.2f} GB | "
        f"slices/plane={slices_per_plane} | "
        f"labels: Low={info['label_distribution']['low_grade']}, High={info['label_distribution']['high_grade']}"
    )
    return slices_array, labels_array, patient_ids_array, metadata_array, feature_names, info


def scale_fold_split(
    slices_all: np.ndarray,
    labels_all: np.ndarray,
    patient_ids_all: np.ndarray,
    fold_train_patients: np.ndarray,
    fold_test_patients: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split and per-channel z-score scale slices for one CV fold.

    Reshapes (N, C, H, W) → (N*H*W, C) so StandardScaler normalises per channel
    (one mean/std per modality), then restores the original shape.
    Scaler is fit on training data only to prevent data leakage.

    Parameters
    ----------
    slices_all : np.ndarray, shape (N, C, H, W)
    labels_all : np.ndarray, shape (N,)
    patient_ids_all : np.ndarray, shape (N,)
    fold_train_patients : np.ndarray  — patient indices in the training split
    fold_test_patients : np.ndarray   — patient indices in the test split

    Returns
    -------
    X_train_scaled, y_train, train_patient_ids,
    X_test_scaled,  y_test,  test_patient_ids,
    scaler : StandardScaler  — fit on training data only, for reproducibility
    """
    train_mask = np.isin(patient_ids_all, fold_train_patients)
    test_mask = np.isin(patient_ids_all, fold_test_patients)

    X_train, y_train = slices_all[train_mask], labels_all[train_mask]
    X_test, y_test = slices_all[test_mask], labels_all[test_mask]

    N_tr, C, H, W = X_train.shape
    N_te = X_test.shape[0]

    # Calculate means and stds across all axes except channels (axis 1)
    means = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    stds = np.std(X_train, axis=(0, 2, 3), keepdims=True)
    
    # Avoid division by zero
    stds[stds == 0] = 1.0

    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds

    # Create dummy scaler for compatibility
    scaler = StandardScaler()
    scaler.mean_ = means.flatten()
    scaler.scale_ = stds.flatten()
    scaler.var_ = stds.flatten() ** 2

    return X_train_scaled, y_train, patient_ids_all[train_mask], X_test_scaled, y_test, patient_ids_all[test_mask], scaler
