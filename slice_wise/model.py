"""Modular CNN backbone for slice-wise tumor grade classification."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoModelForZeroShotImageClassification
import torch.hub
import medim
import tensorflow as tf
from huggingface_hub import snapshot_download
import keras

from ..shared.model import create_model


# Feature dimensions per backbone / extract_layer.
# Spatial layers (layer1-4) return (B, C, H, W) and are AdaptiveAvgPool2d'd before flattening.
_LAYER_DIMS = {
    'resnet18':             {'layer1': 64,  'layer2': 128, 'layer3': 256, 'layer4': 512,  'avgpool': 512,  'fc': 1000},
    'resnet34':             {'layer1': 64,  'layer2': 128, 'layer3': 256, 'layer4': 512,  'avgpool': 512,  'fc': 1000},
    'resnet50':             {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avgpool': 2048, 'fc': 1000},
    'radimagenet_resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avgpool': 2048},
    'efficientnet_b0':      {'avgpool': 1280},
    'efficientnet_b1':      {'avgpool': 1280},
    'medim_stunet_b':       {'conv_blocks_context': 512, 'encoder': 512},
    'medsiglip_448':        {'cls_token': 1152},
    'path_foundation':      {'output': 128},  # requires tensorflow + keras
}


class ChannelAdapter(nn.Module):
    """Learnable 1x1 conv projecting from in_channels to out_channels."""

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
class PretrainedBackbone(nn.Module):
    """Pretrained CNN feature extractor with configurable extraction layer.

    Supported backbones and valid extract_layer values:
        resnet18/34/50, radimagenet_resnet50 : layer1, layer2, layer3, layer4, avgpool, fc
        efficientnet_b0/b1                  : avgpool
        medim_stunet_b                      : encoder
        medsiglip_448                       : cls_token  (HuggingFace transformers, PyTorch)
        path_foundation                     : output     (TF SavedModel, requires tensorflow+keras)

    Spatial outputs (layer1-4) are global-average-pooled before flattening.
    All outputs: (batch, feature_dim).
    """

    _TORCHVISION = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
    }

    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        extract_layer: str = 'avgpool',
        freeze: bool = True,
    ):
        super().__init__()
        assert backbone in _LAYER_DIMS, \
            f"Unknown backbone '{backbone}'. Supported: {list(_LAYER_DIMS)}"
        assert extract_layer in _LAYER_DIMS[backbone], (
            f"Invalid extract_layer '{extract_layer}' for '{backbone}'. "
            f"Valid: {list(_LAYER_DIMS[backbone])}"
        )
        self.backbone_name = backbone
        self.feature_dim = _LAYER_DIMS[backbone][extract_layer]

        if backbone in self._TORCHVISION:
            base = self._TORCHVISION[backbone](pretrained=pretrained)
            self.extractor = create_feature_extractor(base, return_nodes={extract_layer: 'out'})

        elif backbone == 'radimagenet_resnet50':
            base = torch.hub.load('Warvito/radimagenet-models', 'radimagenet_resnet50')
            self.extractor = create_feature_extractor(base, return_nodes={extract_layer: 'out'})

        elif backbone == 'medim_stunet_b':
            model = medim.create_model("STU-Net-B", dataset="BraTS21")
            self.extractor = model.conv_blocks_context

        elif backbone == 'medsiglip_448':
            full = AutoModelForZeroShotImageClassification.from_pretrained("google/medsiglip-448")
            self.extractor = full.vision_model

        elif backbone == 'path_foundation':
            tf.config.set_visible_devices([], 'GPU')
            with tf.device('/CPU:0'):
                self.extractor = keras.layers.TFSMLayer(
                    snapshot_download(repo_id="google/path-foundation"),
                    call_endpoint='serving_default'
                )

        if freeze and backbone != 'path_foundation':
            for p in self.extractor.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == 'path_foundation':
            import tensorflow as tf
            x_np = np.transpose(x.detach().cpu().numpy(), (0, 2, 3, 1))
            out = self.extractor(tf.convert_to_tensor(x_np, dtype=tf.float32))
            feat = list(out.values())[0] if isinstance(out, dict) else out
            feat = torch.from_numpy(feat.numpy()).to(x.device).float()
            return feat.mean(dim=[1, 2]) if feat.ndim == 4 else feat

        elif self.backbone_name == 'medsiglip_448':
            out = self.extractor(pixel_values=x, return_dict=True)
            return out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]

        elif self.backbone_name == 'medim_stunet_b':
            # STUNet expects 5D input (N, C, D, H, W). Add depth D=1.
            x = x.unsqueeze(2)
            for block in self.extractor:
                x = block(x)
            return F.adaptive_avg_pool3d(x, 1).flatten(1)

        else:
            out = self.extractor(x)['out']
            if out.ndim == 4:
                out = F.adaptive_avg_pool2d(out, 1)
            return out.flatten(1)

class ModularTumorClassifier(nn.Module):
    """ChannelAdapter → PretrainedBackbone → sklearn classifier head.

    Backbone starts frozen. Call set_backbone_frozen() to control unfreezing for fine-tuning.
    Classifier is a sklearn-backed nn.Module from shared/model.py.
    """

    def __init__(
        self,
        n_input_channels: int,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        extract_layer: str = 'avgpool',
        freeze_backbone: bool = True,
        adapter_out_channels: int = 3,
        classifier_type: str = 'logistic_regression',
        **classifier_kwargs,
    ):
        super().__init__()
        self.adapter = ChannelAdapter(n_input_channels, adapter_out_channels)
        self.backbone = PretrainedBackbone(backbone, pretrained, extract_layer, freeze=freeze_backbone)
        self.classifier = create_model(classifier_type, **classifier_kwargs)
        self.classifier_type = classifier_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(self.adapter(x)))

    def set_backbone_frozen(self, n_layers_unfrozen: int = 0) -> None:
        """Freeze or partially unfreeze the backbone.

        Parameters
        ----------
        n_layers_unfrozen : int
            0   — freeze all backbone parameters.
            -1  — unfreeze all backbone parameters.
            N>0 — unfreeze the last N top-level parameter-bearing children
                  from the output end (e.g. N=1 → layer4, N=2 → layer3+4 for ResNet).
        """
        assert n_layers_unfrozen >= -1
        extractor = self.backbone.extractor
        for p in extractor.parameters():
            p.requires_grad_(False)

        if n_layers_unfrozen == -1:
            for p in extractor.parameters():
                p.requires_grad_(True)
        elif n_layers_unfrozen > 0:
            blocks = [m for m in extractor.children()
                      if sum(p.numel() for p in m.parameters()) > 0]
            for block in blocks[-n_layers_unfrozen:]:
                for p in block.parameters():
                    p.requires_grad_(True)

    def extract_features(self, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
        """Pass slices through adapter + backbone and return flat feature vectors.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (N, C, H, W).
        device : torch.device
        batch_size : int
            Number of slices processed per forward pass.
        """
        features_list = []
        self.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i : i + batch_size]).to(device)
                batch_features = self.backbone(self.adapter(batch_X)).cpu().numpy()
                features_list.append(batch_features)
                del batch_X, batch_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return np.vstack(features_list)

    def load_phase1_checkpoint(
        self,
        checkpoint_dir: Path,
        fold_idx: int,
        expected_n_channels: int,
        device: torch.device,
        unfreeze_n_layers: int = 0,
    ) -> None:
        """Load a Phase 1 checkpoint and optionally unfreeze backbone layers.

        Parameters
        ----------
        checkpoint_dir : Path
            Directory that contains per-fold checkpoint files.
        fold_idx : int
            0-based fold index; resolves to ``checkpoint_fold_{fold_idx+1}.pth``.
        expected_n_channels : int
            Number of input channels expected in the checkpoint; asserted to match.
        device : torch.device
            Device to map the checkpoint tensors onto.
        unfreeze_n_layers : int
            Passed to :meth:`set_backbone_frozen` after loading; 0 keeps all layers frozen.
        """
        fold_checkpoint = Path(checkpoint_dir) / f"checkpoint_fold_{fold_idx + 1}.pth"
        assert fold_checkpoint.exists(), f"Missing Phase 1 checkpoint: {fold_checkpoint}"

        checkpoint = torch.load(fold_checkpoint, map_location=device)
        checkpoint_n_channels = checkpoint['n_input_channels']
        assert checkpoint_n_channels == expected_n_channels, (
            f"Channel mismatch: checkpoint has {checkpoint_n_channels}, "
            f"current data has {expected_n_channels}"
        )

        self.load_state_dict(checkpoint['model_state_dict'])
        if unfreeze_n_layers > 0:
            self.set_backbone_frozen(n_layers_unfrozen=unfreeze_n_layers)

    def fine_tune(
        self,
        build_batch: Callable[[np.ndarray], np.ndarray],
        y: np.ndarray,
        n_samples: int,
        device: torch.device,
        lr: float,
        num_epochs: int,
        batch_size: int,
        seed: int = 42,
    ) -> list[float]:
        """End-to-end fine-tune adapter + currently-unfrozen backbone stages.

        Attaches a temporary ``nn.Linear(feature_dim, 1)`` head, trains the
        trainable parameters with Adam + BCEWithLogitsLoss for ``num_epochs``
        epochs, then discards the head. After this call the backbone weights
        reflect the fine-tuned state.

        Parameters
        ----------
        build_batch : callable
            ``build_batch(indices) -> np.ndarray`` of shape (B, C, H, W).
            Called on each minibatch with a numpy array of integer indices.
        y : np.ndarray
            Per-sample labels (length ``n_samples``).
        n_samples : int
            Total number of training samples (indices 0..n_samples-1).

        BatchNorm layers stay in eval mode (running stats frozen) — preferred
        for small medical-imaging datasets where batch statistics are noisy.
        """
        assert num_epochs > 0
        assert batch_size > 0
        assert len(y) == n_samples

        head = nn.Linear(self.backbone.feature_dim, 1).to(device)
        trainable = [p for p in self.parameters() if p.requires_grad] + list(head.parameters())
        assert len(trainable) > 0, (
            "fine_tune called but no parameters require gradients — "
            "did you forget to call set_backbone_frozen(n>0)?"
        )
        optimizer = torch.optim.Adam(trainable, lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        self.eval()  # keep BN running stats fixed
        head.train()

        rng = np.random.default_rng(seed)
        epoch_losses: list[float] = []
        for epoch in range(num_epochs):
            perm = rng.permutation(n_samples)
            running = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_np = build_batch(idx)
                batch_x = torch.from_numpy(batch_np).float().to(device)
                batch_y = torch.from_numpy(y[idx]).float().to(device)

                feat = self.backbone(self.adapter(batch_x))
                logits = head(feat).squeeze(-1)
                loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += float(loss.item())
                n_batches += 1
                del batch_x, batch_y, feat, logits, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            epoch_losses.append(running / max(1, n_batches))

        return epoch_losses