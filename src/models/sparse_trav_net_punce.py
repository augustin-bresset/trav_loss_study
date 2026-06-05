"""SparseTravNet wrapper for puNCE contrastive training.

Adds a projection head on top of the existing backbone via a forward hook.
The backbone (SparseTravNet) is never modified — its weights and state dict
are identical to a standalone SparseTravNet, so checkpoints are interchangeable.

Usage::

    backbone = SparseTravNet(in_channels=4, cr=0.5)
    # optionally load a pre-trained checkpoint:
    # backbone.load_state_dict(torch.load("best.pth"))

    model = SparseTravNetPUNCE(backbone, proj_dim=64)
    # model(x) -> (logits (N,), embeddings (N, proj_dim))

    # Save/load only backbone weights (compatible with SparseTravNet):
    torch.save(model.backbone.state_dict(), "best.pth")
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_trav_net import SparseTravNet


class SparseTravNetPUNCE(nn.Module):
    """SparseTravNet + MLP projection head for puNCE training.

    Uses a ``register_forward_pre_hook`` on the backbone's classification
    head to capture per-voxel features without touching the backbone code.

    Args:
        backbone:  A SparseTravNet instance (any cr).
        proj_dim:  Output dimension of the projection head.
    """

    def __init__(self, backbone: SparseTravNet, proj_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone

        feat_dim = backbone.head.in_features   # cs[4] = int(cr * 32)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        self._features: torch.Tensor | None = None
        self._hook_handle = backbone.head.register_forward_pre_hook(self._capture)

    def _capture(self, module: nn.Module, inputs: tuple) -> None:
        """Hook: stores the feature tensor fed into the classification head."""
        self._features = inputs[0]

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, embeddings).

        logits:     (N,)      — raw classification scores (identical to
                               SparseTravNet output, same checkpoint).
        embeddings: (N, proj_dim) — projection head output (NOT normalised;
                               PUNCELoss normalises internally).
        """
        logits = self.backbone(x)                     # triggers hook
        embeddings = self.projector(self._features)   # (N, proj_dim)
        return logits, embeddings

    def __del__(self) -> None:
        self._hook_handle.remove()
