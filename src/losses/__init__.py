import torch.nn as nn

from .losses import ClassificationCriterion
from .focal_loss import BinaryFocalLoss
from .pu_loss import uPULoss, nnPULoss
from .binary_losses import (
    BinaryTverskyLoss,
    BinaryLovaszHingeLoss,
    FocalnnPULoss,
    BinaryAsymmetricLoss,
)

TRAV_LOSSES = {
    # ── Baselines ──────────────────────────────────────────────────────────
    "bce":     lambda cfg: nn.BCEWithLogitsLoss(
                   pos_weight=None if cfg.get("pos_weight") is None
                   else __import__("torch").tensor(cfg["pos_weight"])
               ),
    "focal":   lambda cfg: BinaryFocalLoss(
                   gamma=cfg.get("gamma", 2.0),
                   pos_weight=cfg.get("pos_weight"),
               ),
    "upu":     lambda cfg: uPULoss(prior=cfg.get("prior", 0.1)),
    "nnpu":    lambda cfg: nnPULoss(prior=cfg.get("prior", 0.1), beta=cfg.get("beta", 0.0)),
    # ── New losses ─────────────────────────────────────────────────────────
    "tversky": lambda cfg: BinaryTverskyLoss(
                   alpha=cfg.get("tversky_alpha", 0.3),
                   beta=cfg.get("tversky_beta", 0.7),
               ),
    "lovasz":  lambda cfg: BinaryLovaszHingeLoss(),
    "focal_nnpu": lambda cfg: FocalnnPULoss(
                   prior=cfg.get("prior", 0.3),
                   gamma=cfg.get("gamma", 2.0),
                   beta=cfg.get("beta", 0.0),
               ),
    "asl":     lambda cfg: BinaryAsymmetricLoss(
                   gamma_pos=cfg.get("gamma_pos", 0.0),
                   gamma_neg=cfg.get("gamma_neg", 4.0),
                   clip=cfg.get("asl_clip", 0.05),
               ),
}

__all__ = [
    "ClassificationCriterion",
    "BinaryFocalLoss",
    "uPULoss",
    "nnPULoss",
    "BinaryTverskyLoss",
    "BinaryLovaszHingeLoss",
    "FocalnnPULoss",
    "BinaryAsymmetricLoss",
    "TRAV_LOSSES",
]