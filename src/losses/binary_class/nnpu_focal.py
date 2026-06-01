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




if __name__ == "__main__":
    import unittest

    class TestFocalNNPULoss(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = FocalnnPULoss(prior=0.3, gamma=2.0)
            self.logits  = torch.randn(128)
            self.targets = torch.randint(0, 2, (128,)).long()

        def test_output_scalar(self):
            self.assertEqual(self.loss(self.logits, self.targets).shape, torch.Size([]))

        def test_backward(self):
            logits = self.logits.requires_grad_(True)
            self.loss(logits, self.targets).backward()
            self.assertIsNotNone(logits.grad)

        def test_invalid_prior_raises(self):
            with self.assertRaises(ValueError):
                FocalnnPULoss(prior=0.0)

        def test_no_positives_fallback(self):
            targets_unl = torch.zeros(128, dtype=torch.long)
            out = self.loss(self.logits, targets_unl)
            self.assertEqual(out.shape, torch.Size([]))

        def test_gamma_zero_matches_nnpu(self):
            # gamma=0 → focal degenerates to BCE → should match nnPULoss
            from nnpu_bce import nnPULoss
            loss_focal0 = FocalnnPULoss(prior=0.3, gamma=0.0)
            loss_nnpu   = nnPULoss(prior=0.3)
            out_focal = loss_focal0(self.logits, self.targets.float())
            out_nnpu  = loss_nnpu(self.logits, self.targets)
            self.assertAlmostEqual(out_focal.item(), out_nnpu.item(), places=4)

        def test_nonneg_clamp_activates(self):
            loss_clamped = FocalnnPULoss(prior=0.99, gamma=2.0, beta=0.0)
            out = loss_clamped(self.logits, self.targets)
            self.assertGreaterEqual(out.item(), 0.0)

    unittest.main()
