from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load dataset into standardized DataFrame format.
        
        Returns
        -------
        pd.DataFrame
            Patient-wise table with columns:
            - Imaging modalities (3D/4D numpy arrays)
            - 'GT': ground truth segmentation mask
            - 'grade': binary label (0=low, 1=high)
        """
        pass
    
    @abstractmethod
    def get_available_modalities(self) -> list[str]:
        """Return list of imaging modalities in this dataset."""
        pass