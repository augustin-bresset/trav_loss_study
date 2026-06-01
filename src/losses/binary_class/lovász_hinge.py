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


if __name__ == "__main__":
    import unittest

    class TestBinaryLovaszHingeLoss(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = BinaryLovaszHingeLoss()
            self.logits  = torch.randn(128)
            self.targets = torch.randint(0, 2, (128,)).long()

        def test_output_scalar(self):
            self.assertEqual(self.loss(self.logits, self.targets).shape, torch.Size([]))

        def test_non_negative(self):
            self.assertGreaterEqual(self.loss(self.logits, self.targets).item(), 0.0)

        def test_backward(self):
            logits = self.logits.requires_grad_(True)
            self.loss(logits, self.targets).backward()
            self.assertIsNotNone(logits.grad)

        def test_perfect_predictions_near_zero(self):
            perfect = torch.where(self.targets == 1,
                                  torch.full_like(self.logits,  10.0),
                                  torch.full_like(self.logits, -10.0))
            self.assertLess(self.loss(perfect, self.targets).item(), 0.01)

        def test_degenerate_all_positive_fallback(self):
            # All targets=1 → falls back to BCE, should not crash
            targets_all_pos = torch.ones(128, dtype=torch.long)
            out = self.loss(self.logits, targets_all_pos)
            self.assertGreaterEqual(out.item(), 0.0)

        def test_degenerate_all_negative_fallback(self):
            targets_all_neg = torch.zeros(128, dtype=torch.long)
            out = self.loss(self.logits, targets_all_neg)
            self.assertGreaterEqual(out.item(), 0.0)

    unittest.main()
