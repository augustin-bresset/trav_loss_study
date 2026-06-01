"""Focal Tversky Loss (Abraham & Khan, 2019).

Raises the Tversky index deficit to a power γ to focus training on hard,
misclassified voxels.  γ < 1 up-weights easy examples; γ > 1 down-weights
them (standard usage in segmentation is γ = 4/3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for binary segmentation.

    FTL = (1 - TI)^γ   where   TI = (TP + ε) / (TP + α·FP + β·FN + ε)

    Args:
        alpha:  FP weight in the Tversky denominator.
        beta:   FN weight in the Tversky denominator.
        gamma:  Focusing exponent (default 4/3 as in the original paper).
        smooth: Laplace smoothing constant.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4 / 3,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        targets = targets.float()

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - ti) ** self.gamma

    def __repr__(self) -> str:
        return (
            f"FocalTverskyLoss(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
        )


if __name__ == "__main__":
    import unittest

    class TestFocalTverskyLoss(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = FocalTverskyLoss()
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

        def test_perfect_near_zero(self):
            perfect = torch.where(self.targets == 1,
                                  torch.full_like(self.logits,  10.0),
                                  torch.full_like(self.logits, -10.0))
            self.assertLess(self.loss(perfect, self.targets).item(), 0.01)

        def test_gamma_one_equals_tversky(self):
            from tversky import BinaryTverskyLoss
            loss_tv  = BinaryTverskyLoss(alpha=0.3, beta=0.7, smooth=1.0)
            loss_ftv = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0, smooth=1.0)
            self.assertAlmostEqual(
                loss_tv(self.logits, self.targets).item(),
                loss_ftv(self.logits, self.targets).item(),
                places=5,
            )

    unittest.main()
