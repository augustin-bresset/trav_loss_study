"""Additional binary losses for traversability PU learning.

Motivated by study_goose results: BCE/focal/uPU had precision≈0.87 but
recall≤0.23; nnPU(prior=0.5) fixed recall at 1.0 but dropped precision.
Goal: find losses that balance precision and recall better.

New losses:
  - BinaryTverskyLoss      : Dice generalisation, penalise FN more than FP
  - BinaryLovaszHingeLoss  : differentiable binary IoU (Lovász extension)
  - FocalnnPULoss          : nnPU but with focal as base (hard-example focus)
  - BinaryAsymmetricLoss   : ASL (Ridnik et al. 2021), clips easy negatives
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------------

class BinaryTverskyLoss(nn.Module):
    """Tversky loss — generalises Dice with asymmetric FP/FN weights.

    TI = TP / (TP + alpha*FP + beta*FN)
    Loss = 1 - TI

    Setting alpha < beta penalises false negatives more → boosts recall.
    alpha + beta = 1 is a common convention (Salehi et al. 2017).

    Args:
        alpha:  FP weight in the denominator.
        beta:   FN weight in the denominator.
        smooth: Laplace smoothing constant.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky

    def __repr__(self) -> str:
        return f"BinaryTverskyLoss(alpha={self.alpha}, beta={self.beta})"


# ---------------------------------------------------------------------------
# Lovász-Hinge Loss
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient for a sorted binary ground truth."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-8)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class BinaryLovaszHingeLoss(nn.Module):
    """Binary Lovász-Hinge loss (Berman et al. 2018).

    Directly optimises the binary Jaccard index (IoU) via its Lovász
    extension. More principled than BCE for segmentation tasks where
    the evaluation metric is F1 / IoU.

    Input logits are raw scores (pre-sigmoid).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.sum() == 0 or (1 - targets).sum() == 0:
            return F.binary_cross_entropy_with_logits(logits, targets.float())

        signs = 2.0 * targets.float() - 1.0           # +1 for pos, -1 for neg
        errors = 1.0 - logits * signs                  # hinge errors
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = targets[perm.data]
        grad = _lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def __repr__(self) -> str:
        return "BinaryLovaszHingeLoss()"


# ---------------------------------------------------------------------------
# Focal-nnPU Loss
# ---------------------------------------------------------------------------

def _focal_loss_per_point(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-point focal loss (no reduction)."""
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    return bce * (1 - p_t) ** gamma


class FocalnnPULoss(nn.Module):
    """nnPU risk estimator using focal loss as the base instead of BCE.

    Combines the PU-correction of nnPU with the hard-example focus of
    focal loss. Useful when easy negatives dominate the unlabeled set.

    Args:
        prior: π — estimated fraction of positives among unlabeled.
        gamma: Focal focusing parameter.
        beta:  nnPU floor for the negative risk term.
    """

    def __init__(self, prior: float = 0.3, gamma: float = 2.0, beta: float = 0.0) -> None:
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = targets == 1
        unl = targets == 0

        if pos.sum() == 0:
            return _focal_loss_per_point(logits[unl], torch.zeros_like(logits[unl]), self.gamma).mean()

        f_p_pos = _focal_loss_per_point(logits[pos], torch.ones_like(logits[pos]),  self.gamma).mean()
        f_p_neg = _focal_loss_per_point(logits[pos], torch.zeros_like(logits[pos]), self.gamma).mean()
        f_u_neg = (
            _focal_loss_per_point(logits[unl], torch.zeros_like(logits[unl]), self.gamma).mean()
            if unl.sum() > 0
            else torch.tensor(0.0, device=logits.device)
        )

        neg_risk = f_u_neg - self.prior * f_p_neg

        if neg_risk < self.beta:
            loss = self.prior * f_p_pos - neg_risk.detach() + self.beta
        else:
            loss = self.prior * f_p_pos + neg_risk

        return loss

    def __repr__(self) -> str:
        return f"FocalnnPULoss(prior={self.prior}, gamma={self.gamma}, beta={self.beta})"


# ---------------------------------------------------------------------------
# Asymmetric Loss (ASL)
# ---------------------------------------------------------------------------

class BinaryAsymmetricLoss(nn.Module):
    """Asymmetric Loss for binary classification (Ridnik et al. 2021).

    Addresses positive/negative imbalance by:
      1. Applying different focusing exponents to positives (gamma_pos)
         and negatives (gamma_neg).
      2. Clipping small negative predictions (probability shift by `clip`)
         to zero out easy negatives entirely.

    This effectively removes the easy-negative gradient without requiring
    an explicit pos_weight, making it robust to varying class ratios.

    Args:
        gamma_pos: Focusing exponent for positive examples (default 0).
        gamma_neg: Focusing exponent for negative examples (default 4).
        clip:      Probability margin shift for negatives [0, 1].
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob = torch.sigmoid(logits)

        # Shift and clip negative probabilities to remove easy negatives
        prob_neg = (prob + self.clip).clamp(max=1.0)

        # BCE per point
        loss_pos = -targets       * torch.log(prob     + 1e-8)
        loss_neg = -(1 - targets) * torch.log(1 - prob_neg + 1e-8)

        # Asymmetric focusing
        loss_pos = loss_pos * (1 - prob)     ** self.gamma_pos
        loss_neg = loss_neg * prob_neg        ** self.gamma_neg

        return (loss_pos + loss_neg).mean()

    def __repr__(self) -> str:
        return (
            f"BinaryAsymmetricLoss(gamma_pos={self.gamma_pos}, "
            f"gamma_neg={self.gamma_neg}, clip={self.clip})"
        )
