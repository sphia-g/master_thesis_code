"""Data preparation for voxel-wise classification."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def extract_voxel_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    sample_ratio: float = 0.1,
    max_voxels_per_patient: int | None = None,
    tumor_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract voxel-level features from all patients.

    Parameters
    ----------
    df : pd.DataFrame
        Patient dataframe with imaging modalities, 'GT' mask, and 'grade' column.
    feature_cols : list[str]
        Modality columns to extract features from. 4D volumes (e.g. PET_dynamic)
        are expanded into one feature per timeframe.
    sample_ratio : float
        Fraction of eligible voxels to sample per patient.
    max_voxels_per_patient : int | None
        Hard cap on sampled voxels per patient after sample_ratio is applied.
    tumor_only : bool
        If True, sample only from tumor voxels (GT == 1).
        If False, sample from all voxel coordinates in the volume,
        i.e. tumor and non-tumor regions including surrounding.

    Returns
    -------
    X : np.ndarray, shape (n_voxels, n_features)
    y : np.ndarray, shape (n_voxels,)
    patient_ids : np.ndarray, shape (n_voxels,)  — integer index into df
    feature_names : list[str]  — one name per column, 4D expanded as <col>_t<i>
    """
    # Expand feature names: 4D volumes become <col>_t0, <col>_t1, ...
    feature_names = []
    for col in feature_cols:
        vol = df[col].iloc[0]
        if vol.ndim == 4:
            feature_names.extend(f"{col}_t{t}" for t in range(vol.shape[-1]))
        else:
            feature_names.append(col)

    n_features = len(feature_names)
    assert 0 < sample_ratio <= 1.0, f"sample_ratio must be in (0, 1], got {sample_ratio}"

    voxel_type = "tumor" if tumor_only else "all"
    logger.info(f"Extracting {n_features} features from {voxel_type} voxels ({sample_ratio*100:.1f}% sampled)")

    all_X, all_y, all_pids = [], [], []

    for patient_idx in range(len(df)):
        GT = df['GT'].iloc[patient_idx]
        assert GT.ndim == 3, f"Expected 3D GT mask for patient {patient_idx}, got shape {GT.shape}"

        if tumor_only:
            coords = np.argwhere(GT == 1)
        else:
            coords = np.argwhere(np.ones(GT.shape, dtype=bool))
        n_voxels = len(coords)

        if n_voxels == 0:
            logger.warning(f"Patient {patient_idx}: no {voxel_type} voxels, skipping")
            continue

        n_sample = int(n_voxels * sample_ratio)
        if max_voxels_per_patient:
            n_sample = min(n_sample, max_voxels_per_patient)
        n_sample = max(1, n_sample)

        sampled_coords = coords[np.random.choice(n_voxels, size=n_sample, replace=False)]
        xs, ys, zs = sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]
        
        voxel_features = np.empty((n_sample, n_features), dtype=np.float32)
        feat_idx = 0
        for col in feature_cols:
            vol = df[col].iloc[patient_idx]
            if vol.ndim == 4:
                for t in range(vol.shape[-1]):
                    voxel_features[:, feat_idx] = vol[xs, ys, zs, t]
                    feat_idx += 1
            else:
                voxel_features[:, feat_idx] = vol[xs, ys, zs]
                feat_idx += 1

        valid = ~np.isnan(voxel_features).any(axis=1)
        voxel_features = voxel_features[valid]
        if len(voxel_features) == 0:
            logger.warning(f"Patient {patient_idx}: all sampled voxels are NaN, skipping")
            continue

        label = int(df['grade'].iloc[patient_idx])
        assert label in (0, 1), f"Patient {patient_idx}: unexpected grade value {label!r}"

        n = len(voxel_features)
        all_X.append(voxel_features)
        all_y.append(np.full(n, label, dtype=np.int8))
        all_pids.append(np.full(n, patient_idx, dtype=np.int32))

    assert all_X, "No voxels extracted — check GT masks and feature columns"

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    patient_ids = np.concatenate(all_pids)

    logger.info(f"Extracted {len(X)} voxels from {len(df)} patients | shape {X.shape}")
    logger.info(f"Label distribution: Low={np.sum(y==0)}, High={np.sum(y==1)}")

    return X, y, patient_ids, feature_names

def scale_fold_split(
    X_all: np.ndarray,
    y_all: np.ndarray,
    patient_ids: np.ndarray,
    fold_train_patients: np.ndarray,
    fold_test_patients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Scale data for one CV fold.

    Scaler is fit on fold training voxels only to prevent data leakage.

    Returns
    -------
    X_train_scaled, y_train, train_patient_ids,
    X_test_scaled, y_test, test_patient_ids, scaler
    """
    train_mask = np.isin(patient_ids, fold_train_patients)
    test_mask = np.isin(patient_ids, fold_test_patients)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, patient_ids[train_mask], X_test_scaled, y_test, patient_ids[test_mask], scaler