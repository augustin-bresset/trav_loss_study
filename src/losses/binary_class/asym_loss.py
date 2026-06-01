"""Additional binary losses for traversability PU learning.
  - BinaryAsymmetricLoss   : ASL (Ridnik et al. 2021), clips easy negatives
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
