"""Additional binary losses for traversability PU learning.
  - BinaryAsymmetricLoss   : ASL (Ridnik et al. 2021), clips easy negatives
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..asymmetric import AsymmetricLossOptimized
# ---------------------------------------------------------------------------
# Asymmetric Loss (ASL)
# ---------------------------------------------------------------------------

class BinaryAsymmetricLoss(AsymmetricLossOptimized):
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
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ Take out the criterion of the autocast
        """
        return super().forward(x.float(), y.float())


if __name__ == "__main__":
    import unittest

    class TestBinaryAsymmetricLoss(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = BinaryAsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
            self.logits  = torch.randn(128)
            self.targets = torch.randint(0, 2, (128,)).float()

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

        def test_clip_zero_same_as_focal(self):
            # clip=0 and gamma_neg=gamma_pos=0 should behave like BCE
            loss_bce_like = BinaryAsymmetricLoss(gamma_pos=0.0, gamma_neg=0.0, clip=0.0)
            import torch.nn.functional as F
            expected = F.binary_cross_entropy_with_logits(self.logits, self.targets)
            got = loss_bce_like(self.logits, self.targets)
            self.assertAlmostEqual(got.item(), expected.item(), places=4)

    unittest.main()
