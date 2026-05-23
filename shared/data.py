"""Shared data utilities: patient-level cross-validation splitting."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_cv_folds(
    patient_ids: np.ndarray,
    df_grades: np.ndarray,
    n_folds: int,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (fold_train_patients, fold_test_patients) for stratified K-fold CV.

    Parameters
    ----------
    patient_ids : np.ndarray
        Integer patient indices actually present in the extracted data (output of
        np.unique over the patient_ids returned by extract_*_features). Must be a
        subset of valid row indices into df.
    df_grades : np.ndarray
        Grade labels from df['grade'] for ALL df rows. Indexed by patient_ids to
        obtain the per-patient label vector for stratification.
    n_folds : int
    random_state : int

    Yields
    ------
    fold_train_patients, fold_test_patients : np.ndarray
        Subsets of patient_ids for this fold.
    """
    unique_patients = np.unique(patient_ids)
    patient_labels = df_grades[unique_patients].astype(int)

    assert len(unique_patients) == len(patient_labels)
    assert set(np.unique(patient_labels)) == {0, 1}, (
        f"Expected binary grades (0/1), got {np.unique(patient_labels)}"
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(unique_patients, patient_labels):
        yield unique_patients[train_idx], unique_patients[test_idx]
