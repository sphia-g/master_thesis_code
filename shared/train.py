"""Training utilities for model wrappers."""

from __future__ import annotations

import logging
import numpy as np
import torch
from torch import nn


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device
) -> None:
    """Fit any model wrapper.

    nn.Module wrappers receive tensors (they unpack internally via .cpu().numpy()).
    Plain sklearn-style objects (e.g. TabPFN) receive numpy arrays directly.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Fitting {model.__class__.__name__}...")

    if isinstance(model, nn.Module):
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).to(device)
        model.fit(X_tensor, y_tensor)
    else:
        model.fit(X_train, y_train)

    logger.info(f"{model.__class__.__name__} fitted successfully")