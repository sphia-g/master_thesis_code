"""Utility helpers: logging, determinism, and serialization."""

from __future__ import annotations

import logging
import random
import json
from typing import Any, Dict

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """Set seeds for python, numpy, and torch for reproducibility."""
    assert seed >= 0, "Seed must be non-negative"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_): 
            return bool(obj)
        return super().default(obj)

def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary to a JSON file with support for NumPy types."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)


def log_factory(logger: logging.Logger):
    """Create a logging callable."""
    def log_fn(obj: Dict[str, Any]):
        logger.info(
            " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                for k, v in obj.items()
            )
        )
    return log_fn

def save_results(results: dict, filename: str) -> None:
    """Save results to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
