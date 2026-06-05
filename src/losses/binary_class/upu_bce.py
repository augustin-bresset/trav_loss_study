"""Positive-Unlabeled (PU) learning losses for binary traversability.

In the traversability setting:
  - Positive (P): points the robot drove over  -> labeled 1
  - Unlabeled (U): all other points            -> labeled 0
                   (may be traversable or not — we simply don't know)

Standard BCE treats unlabeled points as negative, which is wrong.
These losses correct for that by modelling the unlabeled distribution
as a mixture:  p(x) = π·p(x|+) + (1-π)·p(x|-)

References:
  du Plessis et al. (2014) - Analysis of learning from positive and unlabeled data
  Kiryo et al. (2017)      - Positive-Unlabeled Learning with Non-Negative Risk Estimator
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bce_pos(logits: torch.Tensor) -> torch.Tensor:
    """Per-point BCE loss with target=1 (positive):  log(1 + e^{-f})."""
    return F.softplus(-logits)


def _bce_neg(logits: torch.Tensor) -> torch.Tensor:
    """Per-point BCE loss with target=0 (negative):  log(1 + e^{f})."""
    return F.softplus(logits)


class uPULoss(nn.Module):
    """Unbiased PU risk estimator (du Plessis et al. 2014).

    R(f) = π · R_P^+ - π · R_P^- + R_U^-

    where R_U^- - π·R_P^- is an unbiased estimate of (1-π)·R_N^-.
    Can go negative when the model overfits, which causes training instability.
    Prefer nnPULoss in practice.

    Args:
        prior: π — estimated fraction of truly positive points among unlabeled.
    """

    def __init__(self, prior: float = 0.1) -> None:
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior = prior

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N,) raw scores (pre-sigmoid).
            targets: (N,) binary labels — 1 = positive, 0 = unlabeled.
        """
        pos = targets == 1
        unl = targets == 0

        if pos.sum() == 0:
            return _bce_neg(logits[unl]).mean()

        r_p_pos = _bce_pos(logits[pos]).mean()
        r_p_neg = _bce_neg(logits[pos]).mean()
        r_u_neg = _bce_neg(logits[unl]).mean() if unl.sum() > 0 else torch.tensor(0.0, device=logits.device)

        return self.prior * r_p_pos - self.prior * r_p_neg + r_u_neg



if __name__ == "__main__":
    import unittest

    class TestUPULoss(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = uPULoss(prior=0.3)
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
                uPULoss(prior=0.0)
            with self.assertRaises(ValueError):
                uPULoss(prior=1.5)

        def test_no_positives_fallback(self):
            targets_unl = torch.zeros(128, dtype=torch.long)
            out = self.loss(self.logits, targets_unl)
            self.assertEqual(out.shape, torch.Size([]))

        def test_can_go_negative(self):
            # uPU (unlike nnPU) is allowed to produce negative risk — just check it runs
            loss_high_prior = uPULoss(prior=0.99)
            out = loss_high_prior(self.logits, self.targets)
            self.assertEqual(out.shape, torch.Size([]))

    unittest.main()
