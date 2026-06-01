"""Hybrid binary losses combining two complementary objectives.

All hybrids share the pattern:  L = L_main + λ · L_reg

  BCEDiceLoss    : BCE + λ · Dice   — adds overlap sensitivity to BCE
  BCELovaszLoss  : BCE + λ · Lovász — BCE pointwise stability + IoU alignment
  FocalDiceLoss  : Focal + λ · Dice — hard-example focus + overlap coverage
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .lovász_hinge import BinaryLovaszHingeLoss
    from .focal_loss import BinaryFocalLoss
except ImportError:
    from lovász_hinge import BinaryLovaszHingeLoss  # type: ignore
    from focal_loss import BinaryFocalLoss            # type: ignore


def _binary_dice(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for 1-D binary tensors."""
    targets = targets.float()
    tp = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    return 1.0 - (2.0 * tp + smooth) / (denom + smooth)


class BCEDiceLoss(nn.Module):
    """BCE + λ · Dice.

    Combines pixel-wise BCE cross-entropy with region-based Dice.
    Robust to class imbalance while keeping stable gradients.

    Args:
        dice_weight: λ weight on the Dice term (default 1.0).
        pos_weight:  BCE positive-class weight.
    """

    def __init__(self, dice_weight: float = 1.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self._pw = (
            torch.tensor(pos_weight) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self._pw.to(logits.device) if self._pw is not None else None
        bce  = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)
        dice = _binary_dice(torch.sigmoid(logits), targets)
        return bce + self.dice_weight * dice

    def __repr__(self) -> str:
        return f"BCEDiceLoss(dice_weight={self.dice_weight})"


class BCELovaszLoss(nn.Module):
    """BCE + λ · Lovász-Hinge.

    Combines pointwise BCE with the differentiable IoU surrogate.
    BCE provides stable early training; Lovász drives the model toward
    high IoU at convergence.

    Args:
        lovasz_weight: λ weight on the Lovász term (default 1.0).
        pos_weight:    BCE positive-class weight.
    """

    def __init__(self, lovasz_weight: float = 1.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self._lovasz = BinaryLovaszHingeLoss()
        self._pw = (
            torch.tensor(pos_weight) if pos_weight is not None else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self._pw.to(logits.device) if self._pw is not None else None
        bce    = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)
        lovasz = self._lovasz(logits, targets)
        return bce + self.lovasz_weight * lovasz

    def __repr__(self) -> str:
        return f"BCELovaszLoss(lovasz_weight={self.lovasz_weight})"


class FocalDiceLoss(nn.Module):
    """Focal + λ · Dice.

    Focal handles hard-example imbalance; Dice improves recall on
    rare traversable regions that focal still misses.

    Args:
        gamma:       Focal focusing parameter.
        dice_weight: λ weight on the Dice term (default 1.0).
    """

    def __init__(self, gamma: float = 2.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self._focal = BinaryFocalLoss(gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self._focal(logits, targets)
        dice  = _binary_dice(torch.sigmoid(logits), targets)
        return focal + self.dice_weight * dice

    def __repr__(self) -> str:
        return f"FocalDiceLoss(gamma={self._focal.gamma}, dice_weight={self.dice_weight})"


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from lovász_hinge import BinaryLovaszHingeLoss as _BinaryLovaszHingeLoss
    from focal_loss   import BinaryFocalLoss as _BinaryFocalLoss
    import unittest

    class _HybridTestBase:
        loss_cls = None
        loss_kwargs: dict = {}

        def setUp(self):
            torch.manual_seed(0)
            self.loss    = self.loss_cls(**self.loss_kwargs)
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

        def test_perfect_near_zero(self):
            perfect = torch.where(self.targets == 1,
                                  torch.full_like(self.logits,  10.0),
                                  torch.full_like(self.logits, -10.0))
            self.assertLess(self.loss(perfect, self.targets).item(), 0.05)

    class TestBCEDice(_HybridTestBase, unittest.TestCase):
        loss_cls = BCEDiceLoss

    class TestBCELovasz(_HybridTestBase, unittest.TestCase):
        loss_cls = BCELovaszLoss

    class TestFocalDice(_HybridTestBase, unittest.TestCase):
        loss_cls = FocalDiceLoss

    unittest.main()
