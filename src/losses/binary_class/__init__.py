import torch.nn as nn

from .asym_loss      import BinaryAsymmetricLoss
from .focal_loss     import BinaryFocalLoss
from .focal_tversky  import FocalTverskyLoss
from .hybrid         import BCEDiceLoss, BCELovaszLoss, FocalDiceLoss
from .lovász_hinge   import BinaryLovaszHingeLoss
from .nnpu_bce       import nnPULoss as BCEnnPULoss
from .nnpu_focal     import FocalnnPULoss
from .punce          import PUNCELoss
from .sce_gce        import SymmetricCrossEntropy, GeneralizedCrossEntropy
from .tversky        import BinaryTverskyLoss
from .upu_bce        import uPULoss as BCEuPULoss


TRAV_LOSSES = {
    # ── Baselines ─────────────────────────────────────────────────────────
    "bce":        lambda cfg: nn.BCEWithLogitsLoss(
                      pos_weight=None if cfg.get("pos_weight") is None
                      else __import__("torch").tensor(cfg["pos_weight"])
                  ),
    # ── Imbalance / hard-example focus ────────────────────────────────────
    "focal":      lambda cfg: BinaryFocalLoss(
                      gamma=cfg.get("gamma", 2.0),
                      pos_weight=cfg.get("pos_weight"),
                  ),
    "asl":        lambda cfg: BinaryAsymmetricLoss(
                      gamma_pos=cfg.get("gamma_pos", 0.0),
                      gamma_neg=cfg.get("gamma_neg", 4.0),
                      clip=cfg.get("asl_clip", 0.05),
                  ),
    # ── Overlap / segmentation-metric losses ──────────────────────────────
    "tversky":    lambda cfg: BinaryTverskyLoss(
                      alpha=cfg.get("tversky_alpha", 0.3),
                      beta=cfg.get("tversky_beta", 0.7),
                  ),
    "focal_tversky": lambda cfg: FocalTverskyLoss(
                      alpha=cfg.get("tversky_alpha", 0.3),
                      beta=cfg.get("tversky_beta", 0.7),
                      gamma=cfg.get("ft_gamma", 4 / 3),
                  ),
    "lovasz":     lambda cfg: BinaryLovaszHingeLoss(),
    # ── Hybrid ────────────────────────────────────────────────────────────
    "bce_dice":   lambda cfg: BCEDiceLoss(
                      dice_weight=cfg.get("dice_weight", 1.0),
                      pos_weight=cfg.get("pos_weight"),
                  ),
    "bce_lovasz": lambda cfg: BCELovaszLoss(
                      lovasz_weight=cfg.get("lovasz_weight", 1.0),
                      pos_weight=cfg.get("pos_weight"),
                  ),
    "focal_dice": lambda cfg: FocalDiceLoss(
                      gamma=cfg.get("gamma", 2.0),
                      dice_weight=cfg.get("dice_weight", 1.0),
                  ),
    # ── Noise-robust ──────────────────────────────────────────────────────
    "sce":        lambda cfg: SymmetricCrossEntropy(
                      alpha=cfg.get("sce_alpha", 0.1),
                      beta=cfg.get("sce_beta", 1.0),
                  ),
    "gce":        lambda cfg: GeneralizedCrossEntropy(
                      q=cfg.get("gce_q", 0.7),
                  ),
    # ── PU / weak supervision ─────────────────────────────────────────────
    "bce_upu":    lambda cfg: BCEuPULoss(prior=cfg.get("prior", 0.1)),
    "bce_nnpu":   lambda cfg: BCEnnPULoss(
                      prior=cfg.get("prior", 0.1),
                      beta=cfg.get("beta", 0.0),
                  ),
    "focal_nnpu": lambda cfg: FocalnnPULoss(
                      prior=cfg.get("prior", 0.3),
                      gamma=cfg.get("gamma", 2.0),
                      beta=cfg.get("beta", 0.0),
                  ),
    # ── Contrastive PU ────────────────────────────────────────────────────
    # Requires SparseTravNetPUNCE (wrapper model) — returns (logits, embeddings)
    "punce":      lambda cfg: PUNCELoss(
                      prior=cfg.get("prior", 0.3),
                      temperature=cfg.get("temperature", 0.5),
                      cls_loss=BCEnnPULoss(
                          prior=cfg.get("prior", 0.3),
                          beta=cfg.get("beta", 0.0),
                      ),
                      cls_weight=cfg.get("cls_weight", 1.0),
                      max_voxels=cfg.get("max_voxels", 2048),
                  ),
}

__all__ = [
    "BinaryAsymmetricLoss",
    "BinaryFocalLoss",
    "BinaryLovaszHingeLoss",
    "BinaryTverskyLoss",
    "BCEDiceLoss",
    "BCELovaszLoss",
    "BCEnnPULoss",
    "BCEuPULoss",
    "FocalDiceLoss",
    "FocalnnPULoss",
    "FocalTverskyLoss",
    "GeneralizedCrossEntropy",
    "PUNCELoss",
    "SymmetricCrossEntropy",
    "TRAV_LOSSES",
]
