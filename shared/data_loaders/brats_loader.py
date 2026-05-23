"""BraTS dataset loader - handles NIfTI files from the MICCAI BraTS 2019 dataset."""

from __future__ import annotations

import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Optional

from .base import DataLoader

logger = logging.getLogger(__name__)


class BraTSLoader(DataLoader):
    """Loader for BraTS 2019 dataset stored as NIfTI files."""
    
    # Modality filename suffixes
    MODALITIES = {
        'flair': '_flair.nii',
        't1': '_t1.nii',
        't1ce': '_t1ce.nii',
        't2': '_t2.nii',
    }
    TUMOR_LABELS = [1, 2, 4]  # Tumor segmentation classes
    
    def __init__(self, brats_dir: Path):
        self.brats_dir = Path(brats_dir)
        assert self.brats_dir.exists(), f"BraTS directory not found: {self.brats_dir}"
    
    def load(self, max_patients: Optional[int] = None) -> pd.DataFrame:
        # Use instance attribute if no argument provided and attribute exists
        if max_patients is None and hasattr(self, 'max_patients'):
            max_patients = self.max_patients
        
        logger.info(f"Loading BraTS dataset from {self.brats_dir}")

        # Collect all patient info — grade is encoded in the folder name
        all_patient_info = []
        for grade_folder, grade in [('HGG', 1), ('LGG', 0)]:
            grade_dir = self.brats_dir / grade_folder
            if not grade_dir.exists():
                logger.warning(f"Directory not found: {grade_dir}, skipping")
                continue

            for patient_dir in sorted(grade_dir.iterdir()):
                if not patient_dir.is_dir():
                    continue
                all_patient_info.append({
                    'patient_dir': patient_dir,
                    'patient_id': patient_dir.name,
                    'grade': grade
                })
        
        total_available = len(all_patient_info)
        logger.info(f"Found {total_available} total patients")
        
        # Balanced sampling: fill 50/50 as far as possible, then use remaining budget
        if max_patients and max_patients < total_available:
            hgg_patients = [p for p in all_patient_info if p['grade'] == 1]
            lgg_patients = [p for p in all_patient_info if p['grade'] == 0]

            # Take up to half from each class; minority class sets the ceiling
            n_lgg = min(len(lgg_patients), max_patients // 2)
            n_hgg = min(len(hgg_patients), max_patients - n_lgg)

            random.seed(42)
            all_patient_info = random.sample(hgg_patients, n_hgg) + random.sample(lgg_patients, n_lgg)

            logger.info(f"Selected {n_hgg} HGG + {n_lgg} LGG = {len(all_patient_info)} patients")
        
        # Load the selected patients
        patients_data = []
        for i, patient_info in enumerate(all_patient_info):
            try:
                patient_data = self._load_patient(
                    patient_info['patient_dir'],
                    patient_info['patient_id'],
                    patient_info['grade']
                )
                patients_data.append(patient_data)
                
                # Logging every 50 patients
                if (i + 1) % 50 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(all_patient_info)} patients loaded")
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to load patient {patient_info['patient_id']}: {e}")
                continue
        
        df = pd.DataFrame(patients_data)
        df.set_index('patient_id', inplace=True)
        
        logger.info(f"✓ Loaded {len(df)} patients from BraTS dataset")
        logger.info(f"Grade distribution: LGG={np.sum(df['grade']==0)}, HGG={np.sum(df['grade']==1)}")
        
        return df
    
    def _load_patient(self, patient_dir: Path, patient_id: str, grade: int) -> dict:
        """Load all modalities and segmentation for one patient."""
        patient_data = {'patient_id': patient_id, 'grade': grade}
        
        # Load each modality with memory mapping (doesn't load into RAM immediately)
        for modality_name, suffix in self.MODALITIES.items():
            filepath = patient_dir / f"{patient_id}{suffix}"
            
            if not filepath.exists():
                raise FileNotFoundError(f"Missing {modality_name} file: {filepath}")
            
            nii = nib.load(str(filepath))
            # Use float32 instead of float64 (50% memory reduction!)
            volume = nii.get_fdata(dtype=np.float32)
            patient_data[modality_name] = volume
        
        # Load segmentation
        seg_filepath = patient_dir / f"{patient_id}_seg.nii"
        if not seg_filepath.exists():
            raise FileNotFoundError(f"Missing segmentation file: {seg_filepath}")
        
        seg_nii = nib.load(str(seg_filepath))
        seg_volume = seg_nii.get_fdata(dtype=np.float32)  # Also float32
        
        # Convert multi-class to binary: any tumor class → 1
        GT = np.isin(seg_volume, self.TUMOR_LABELS).astype(np.float32)  # float32!
        patient_data['GT'] = GT
        
        return patient_data
    
    def get_available_modalities(self) -> list[str]:
        return list(self.MODALITIES.keys())