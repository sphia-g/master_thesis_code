"""Naive40 dataset loader - handles .mat file format for the Naive40 dataset."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import mat73

from .base import DataLoader

logger = logging.getLogger(__name__)

class Naive40Loader(DataLoader):
    """Loader for Naive40 dataset stored in .mat format."""
    
    def __init__(self, mat_file_path: Path):
        self.mat_file_path = Path(mat_file_path)
        assert self.mat_file_path.exists(), f"Mat file not found: {self.mat_file_path}"
    
    def load(self) -> pd.DataFrame:
        """Load Naive40 data from .mat file and standardize format.
        
        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with columns:
            - Imaging modalities (3D/4D numpy arrays)
            - 'GT': ground truth segmentation mask (3D array)
            - 'grade': binary label (0=low grade, 1=high grade)
        """
        logger.info(f"Loading Naive40 dataset from {self.mat_file_path}")
        
        d = mat73.loadmat(str(self.mat_file_path))
        df = pd.DataFrame(d['Tcell'], columns=[str(v) for v in d['VarNames']])
        
        if 'RowNames' in d and d['RowNames'] and len(d['RowNames']) > 0:
            df.index = [str(r) for r in d['RowNames']]
        
        # Standardize 4D volumes by truncating timeframes logically
        df = self._standardize_4d_volumes(df)
        
        # Binary grade extraction
        assert 'Grade_Bin' in df.columns, f"'Grade_Bin' col missing. Available: {df.columns.tolist()}"
        df['grade'] = df['Grade_Bin'].astype(int)
        assert set(df['grade'].unique()).issubset({0, 1}), "Grades must be binary (0 or 1)"
        
        # Drop non-imaging columns to prevent accidental inclusion efficiently
        df.drop(columns=['Grade', 'Grade_Bin', 'WHO_Grade'], errors='ignore', inplace=True)
        
        logger.info(f"Loaded {len(df)} patients from Naive40 dataset")
        logger.info(f"Grade distribution: Low={np.sum(df['grade']==0)}, High={np.sum(df['grade']==1)}")
        
        return df

    def _standardize_4d_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 4D volumes have consistent timeframes by truncating to the minimum found."""
        for col in self.get_available_modalities():
            if col not in df.columns:
                continue
                
            # Filter rows that have a 4D array for this modality (e.g., PET_dynamic)
            shapes = [x.shape for x in df[col] if isinstance(x, np.ndarray) and x.ndim == 4]
            if not shapes:
                continue
                
            min_t = min(s[-1] for s in shapes)
            max_t = max(s[-1] for s in shapes)
            
            if min_t != max_t:
                logger.warning(f"Modality '{col}': Inconsistent timeframes ({min_t}-{max_t}). Truncating to {min_t}.")
                # Truncate all 4D volumes efficiently (ignore non-4D just in case)
                df[col] = df[col].apply(lambda x: x[..., :min_t] if isinstance(x, np.ndarray) and x.ndim == 4 else x)
            else:
                logger.info(f"Modality '{col}': All patients consistent with {min_t} timeframes.")
                
        return df

    def get_available_modalities(self) -> list[str]:
        """Return list of imaging modalities available in Naive40 dataset."""
        return [
            'ADC',
            'ADC_K',
            'Darkfluid',
            'PET_late',
            'T1_mprage',
            'T1_mprageKM',
            'T2',
            'PET_dynamic',  # 4D volume (has timeframes)
            'PETwashinslope',
            'PETwashin_intercept',
            'PETwashoutslope',
            'PETwashout_intercept',
            'PETIntegral',
            'PET_TTP',
        ]
