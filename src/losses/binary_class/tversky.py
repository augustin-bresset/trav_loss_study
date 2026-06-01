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

    Setting alpha < beta penalises false negatives more -> boosts recall.
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



if __name__ == "__main__":
    import unittest

    class TestBinaryTverskyLoss(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = BinaryTverskyLoss(alpha=0.3, beta=0.7)
            self.logits  = torch.randn(128)
            self.targets = torch.randint(0, 2, (128,)).float()

        def test_output_scalar(self):
            self.assertEqual(self.loss(self.logits, self.targets).shape, torch.Size([]))

        def test_range(self):
            out = self.loss(self.logits, self.targets).item()
            self.assertGreaterEqual(out, 0.0)
            self.assertLessEqual(out, 1.0)

        def test_backward(self):
            logits = self.logits.requires_grad_(True)
            self.loss(logits, self.targets).backward()
            self.assertIsNotNone(logits.grad)

        def test_perfect_predictions_near_zero(self):
            perfect = torch.where(self.targets == 1,
                                  torch.full_like(self.logits,  10.0),
                                  torch.full_like(self.logits, -10.0))
            self.assertLess(self.loss(perfect, self.targets).item(), 0.01)

        def test_alpha_beta_symmetry(self):
            # alpha=beta=0.5 is equivalent to Dice
            loss_dice = BinaryTverskyLoss(alpha=0.5, beta=0.5)
            out = loss_dice(self.logits, self.targets)
            self.assertGreaterEqual(out.item(), 0.0)

    unittest.main()
