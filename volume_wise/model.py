"""Volume-wise 3D classifier: ChannelAdapter3D → PretrainedBackbone3D → classifier head."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.medicalnet import medicalnet
from ..shared.model import create_model

logger = logging.getLogger(__name__)

# backbone name → {extract_layer: feature_dim}
# BasicBlock (expansion=1): medicalnet 10/18/34
# Bottleneck  (expansion=4): medicalnet 50/101/152/200
_LAYER_DIMS = {
    'medicalnet_10':  {'layer1': 64,  'layer2': 128, 'layer3': 256, 'layer4': 512,  'avgpool': 512},
    'medicalnet_18':  {'layer1': 64,  'layer2': 128, 'layer3': 256, 'layer4': 512,  'avgpool': 512},
    'medicalnet_34':  {'layer1': 64,  'layer2': 128, 'layer3': 256, 'layer4': 512,  'avgpool': 512},
    'medicalnet_50':  {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avgpool': 2048},
    'medicalnet_101': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avgpool': 2048},
    'medicalnet_152': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avgpool': 2048},
    'medicalnet_200': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avgpool': 2048},
}

_MEDICALNET_FACTORY = {
    'medicalnet_10':  (medicalnet.resnet10,  'B'),
    'medicalnet_18':  (medicalnet.resnet18,  'A'),
    'medicalnet_34':  (medicalnet.resnet34,  'A'),
    'medicalnet_50':  (medicalnet.resnet50,  'B'),
    'medicalnet_101': (medicalnet.resnet101, 'B'),
    'medicalnet_152': (medicalnet.resnet152, 'B'),
    'medicalnet_200': (medicalnet.resnet200, 'B'),
}

# Required adapter_out_channels for each inject_layer value.
# inject_layer='layer1' means the stem (conv1/bn1/relu/maxpool) is skipped and the
# adapter output feeds directly into layer1, so channels must equal the stem output (64).
_INJECT_DIMS = {
    'medicalnet_10':  {'stem': 1, 'layer1': 64, 'layer2': 64,  'layer3': 128},
    'medicalnet_18':  {'stem': 1, 'layer1': 64, 'layer2': 64,  'layer3': 128},
    'medicalnet_34':  {'stem': 1, 'layer1': 64, 'layer2': 64,  'layer3': 128},
    'medicalnet_50':  {'stem': 1, 'layer1': 64, 'layer2': 256, 'layer3': 512},
    'medicalnet_101': {'stem': 1, 'layer1': 64, 'layer2': 256, 'layer3': 512},
    'medicalnet_152': {'stem': 1, 'layer1': 64, 'layer2': 256, 'layer3': 512},
    'medicalnet_200': {'stem': 1, 'layer1': 64, 'layer2': 256, 'layer3': 512},
}

# Ordered front-to-back for inject/extract sequencing validation.
_STAGE_ORDER = ['stem', 'layer1', 'layer2', 'layer3', 'layer4']


class ChannelAdapter3D(nn.Module):
    """Learnable 1×1×1 conv projecting from in_channels to out_channels."""

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PretrainedBackbone3D(nn.Module):
    """MedicalNet ResNet3D encoder with configurable injection and extraction layers.

    inject_layer : where the adapter output is fed into the backbone.
        'stem'   (default) — enters conv1; adapter_out_channels must be 1.
        'layer1'           — skips stem;         adapter_out_channels must be 64.
        'layer2'           — skips stem+layer1;  adapter_out_channels must be 64 (BasicBlock) or 256 (Bottleneck).
        'layer3'           — skips stem–layer2;  adapter_out_channels must be 128 (BasicBlock) or 512 (Bottleneck).
        Required values per backbone are listed in _INJECT_DIMS.

    extract_layer : where the feature vector is taken from.
        'layer1'–'layer4' / 'avgpool' — spatial output is global-average-pooled → (B, feature_dim).
        extract_layer must be at or after inject_layer in the network.

    All outputs: (B, feature_dim).
    """

    def __init__(
        self,
        backbone: str,
        pretrained_path: str | None,
        pretrained: bool = True,
        extract_layer: str = 'avgpool',
        inject_layer: str = 'stem',
        freeze: bool = True,
    ):
        super().__init__()
        assert backbone in _LAYER_DIMS, \
            f"Unknown backbone '{backbone}'. Supported: {list(_LAYER_DIMS)}"
        assert extract_layer in _LAYER_DIMS[backbone], (
            f"Invalid extract_layer '{extract_layer}' for '{backbone}'. "
            f"Valid: {list(_LAYER_DIMS[backbone])}"
        )
        assert inject_layer in _INJECT_DIMS[backbone], (
            f"Invalid inject_layer '{inject_layer}'. Valid: {list(_INJECT_DIMS[backbone])}"
        )
        # Normalise 'avgpool' → 'layer4' for ordering comparison
        extract_stage = 'layer4' if extract_layer == 'avgpool' else extract_layer
        assert _STAGE_ORDER.index(inject_layer) <= _STAGE_ORDER.index(extract_stage), (
            f"inject_layer '{inject_layer}' is after extract_layer '{extract_layer}'"
        )

        self.backbone_name = backbone
        self.extract_layer = extract_layer
        self.inject_layer = inject_layer
        self.feature_dim = _LAYER_DIMS[backbone][extract_layer]

        factory, shortcut = _MEDICALNET_FACTORY[backbone]
        net = factory(sample_input_D=1, sample_input_H=1, sample_input_W=1,
                      num_seg_classes=2, shortcut_type=shortcut, no_cuda=True)

        if pretrained:
            assert pretrained_path is not None and len(pretrained_path) > 0, (
                "pretrained=True requires a non-empty pretrained_path"
            )
            path = Path(pretrained_path)
            assert path.exists(), f"Pretrained checkpoint not found: {path}"

            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            state = {k.replace('module.', ''): v for k, v in state.items()
                     if not k.startswith('conv_seg')}
            missing, _ = net.load_state_dict(state, strict=False)
            non_seg_missing = [k for k in missing if not k.startswith('conv_seg')]
            if non_seg_missing:
                logger.warning(f"Missing backbone keys: {non_seg_missing}")
            logger.info(f"Loaded {backbone} from {path} ({len(state)} tensors)")
        else:
            logger.info(f"Using randomly initialized {backbone} backbone (pretrained=False)")

        self.net = net
        if freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder from inject_layer to extract_layer; GAP; return (B, feature_dim)."""
        n = self.net
        extract_stage = 'layer4' if self.extract_layer == 'avgpool' else self.extract_layer

        if self.inject_layer == 'stem':
            x = n.conv1(x); x = n.bn1(x); x = n.relu(x); x = n.maxpool(x)

        for stage_name in ('layer1', 'layer2', 'layer3', 'layer4'):
            if _STAGE_ORDER.index(stage_name) < _STAGE_ORDER.index(self.inject_layer):
                continue  # stage is before injection point — skip
            x = getattr(n, stage_name)(x)
            if stage_name == extract_stage:
                return F.adaptive_avg_pool3d(x, 1).flatten(1)

        assert False, f"extract_layer '{self.extract_layer}' was not reached — this is a bug"


class VolumeWiseClassifier(nn.Module):
    """ChannelAdapter3D → PretrainedBackbone3D → classifier head.

    Backbone starts frozen. Call set_backbone_frozen() to control unfreezing for fine-tuning.
    Mirrors slice_wise.model.ModularTumorClassifier, adapted for 3D volumes.
    """

    def __init__(
        self,
        n_input_channels: int,
        backbone: str = 'medicalnet_10',
        pretrained_path: str | None = None,
        pretrained: bool = True,
        extract_layer: str = 'avgpool',
        inject_layer: str = 'stem',
        adapter_out_channels: int = 1,
        freeze_backbone: bool = True,
        classifier_type: str = 'logistic_regression',
        **classifier_kwargs,
    ):
        super().__init__()
        assert adapter_out_channels == _INJECT_DIMS[backbone][inject_layer], (
            f"adapter_out_channels={adapter_out_channels} does not match the required "
            f"input channels for inject_layer='{inject_layer}' on '{backbone}': "
            f"{_INJECT_DIMS[backbone][inject_layer]}"
        )
        self.adapter = ChannelAdapter3D(n_input_channels, adapter_out_channels)
        self.backbone = PretrainedBackbone3D(
            backbone=backbone,
            pretrained_path=pretrained_path,
            pretrained=pretrained,
            extract_layer=extract_layer,
            inject_layer=inject_layer,
            freeze=freeze_backbone,
        )
        self.classifier = create_model(classifier_type, **classifier_kwargs)
        self.classifier_type = classifier_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(self.adapter(x)))

    def set_backbone_frozen(self, n_layers_unfrozen: int = 0) -> None:
        """Freeze or partially unfreeze the backbone.

        Parameters
        ----------
        n_layers_unfrozen : int
            0   — freeze all backbone parameters (default, Phase 1).
            -1  — unfreeze all backbone parameters.
            N>0 — unfreeze the last N residual stages from the output end
                  (1 → layer4, 2 → layer3+4, 3 → layer2+3+4, 4 → all stages).
        """
        assert n_layers_unfrozen >= -1
        net = self.backbone.net
        for p in net.parameters():
            p.requires_grad_(False)
        if n_layers_unfrozen == -1:
            for p in net.parameters():
                p.requires_grad_(True)
        elif n_layers_unfrozen > 0:
            stages = [net.layer1, net.layer2, net.layer3, net.layer4]
            for stage in stages[-n_layers_unfrozen:]:
                for p in stage.parameters():
                    p.requires_grad_(True)

    def extract_features(
        self,
        X: np.ndarray,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        """Extract backbone features for a batchable volume array.

        Parameters
        ----------
        X : array-like
            Input volumes with shape (N, C, D, H, W).
        device : torch.device
        batch_size : int
            Number of volumes processed per forward pass.
        """
        features = []
        self.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x = torch.FloatTensor(X[i : i + batch_size]).to(device)
                batch_feat = self.backbone(self.adapter(batch_x)).cpu().numpy()
                features.append(batch_feat)
                del batch_x, batch_feat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return np.vstack(features)

    def load_phase1_checkpoint(
        self,
        checkpoint_dir: Path,
        fold_idx: int,
        expected_n_channels: int,
        device: torch.device,
        unfreeze_n_layers: int = 0,
    ) -> None:
        """Load a Phase 1 checkpoint and optionally unfreeze backbone stages.

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
            Passed to :meth:`set_backbone_frozen` after loading; 0 keeps all frozen.
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
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device,
        lr: float,
        num_epochs: int,
        batch_size: int,
        seed: int = 42,
    ) -> list[float]:
        """End-to-end fine-tune adapter + currently-unfrozen backbone stages.

        Attaches a temporary ``nn.Linear(feature_dim, 1)`` head, trains the
        trainable parameters with Adam + BCEWithLogitsLoss for ``num_epochs``
        epochs over ``X``, then discards the head. After this call the backbone
        weights reflect the fine-tuned state and ``extract_features`` will
        produce features from the updated backbone.

        BatchNorm layers stay in eval mode (running stats frozen) — preferred
        for small medical-imaging datasets where batch statistics are noisy.
        """
        assert num_epochs > 0
        assert batch_size > 0

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
        n = len(X)
        epoch_losses: list[float] = []
        for epoch in range(num_epochs):
            perm = rng.permutation(n)
            running = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch_x = torch.from_numpy(X[idx]).float().to(device)
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