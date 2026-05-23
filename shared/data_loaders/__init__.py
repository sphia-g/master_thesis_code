"""Data loaders for different datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .base import DataLoader
from .naive40_loader import Naive40Loader
from .brats_loader import BraTSLoader


def get_loader(
    dataset_name: str,
    dataset_config: dict,
    data_root: Path
) -> DataLoader:
    """Factory function to create appropriate data loader.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset ('naive32' or 'brats')
    dataset_config : dict
        Dataset-specific configuration from config.yaml
    data_root : Path
        Root directory containing data files
    
    Returns
    -------
    DataLoader
        Appropriate loader instance for the dataset
    
    Examples
    --------
    >>> loader = get_loader('naive40', {'mat_file': 'Naive40_export.mat'}, Path('Data/'))
    >>> df = loader.load()
    """
    if dataset_name == 'naive40':
        mat_file = dataset_config['mat_file']
        mat_file_path = data_root / mat_file
        loader = Naive40Loader(mat_file_path)
    
    elif dataset_name == 'brats':
        brats_dir_path = data_root / dataset_config['data_dir']
        loader = BraTSLoader(brats_dir_path)
    
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Supported datasets: 'naive32', ''naive40', 'brats'"
        )
    
    # Store max_patients on loader instance for use in .load()
    loader.max_patients = dataset_config.get('max_patients', None)
    return loader


__all__ = ['DataLoader', 'Naive32Loader', 'Naive40Loader', 'BraTSLoader', 'get_loader']