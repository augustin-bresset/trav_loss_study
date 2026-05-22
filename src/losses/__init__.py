import torch.nn as nn

from .losses import ClassificationCriterion
from .focal_loss import BinaryFocalLoss
from .pu_loss import uPULoss, nnPULoss

TRAV_LOSSES = {
    "bce":   lambda cfg: nn.BCEWithLogitsLoss(),
    "focal": lambda cfg: BinaryFocalLoss(gamma=cfg.get("gamma", 2.0), pos_weight=cfg.get("pos_weight")),
    "upu":   lambda cfg: uPULoss(prior=cfg.get("prior", 0.1)),
    "nnpu":  lambda cfg: nnPULoss(prior=cfg.get("prior", 0.1), beta=cfg.get("beta", 0.0)),
}

__all__ = [
    "ClassificationCriterion",
    "BinaryFocalLoss",
    "uPULoss",
    "nnPULoss",
    "TRAV_LOSSES",
]