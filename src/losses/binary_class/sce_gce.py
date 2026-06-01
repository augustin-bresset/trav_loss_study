"""Noise-robust binary losses.

Both losses are designed for settings where labels are noisy or weakly
supervised (e.g. derived from robot trajectories rather than human annotation).

  SymmetricCrossEntropy (SCE) — Wang et al. (2019)
    L_SCE = α · CE + β · RCE
    RCE penalises confident wrong predictions independently of the label,
    making the loss robust to symmetric label noise.

  GeneralizedCrossEntropy (GCE) — Zhang et al. (2018)
    L_GCE = (1 − p_y^q) / q,  q ∈ (0, 1]
    Interpolates between CE (q→0) and MAE (q=1).  MAE is noise-robust
    because it does not heavily penalise confidently wrong predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCrossEntropy(nn.Module):
    """Symmetric Cross Entropy (Wang et al. 2019).

    L_SCE = α · CE(logits, y) + β · RCE(logits, y)

    where RCE = − Σ p(x) · log(y + ε)  clips the label into (ε, 1−ε)
    before computing the reverse cross-entropy to avoid log(0).

    Args:
        alpha: Weight on the standard CE term.
        beta:  Weight on the reverse CE term.
        eps:   Clipping value for log stability in RCE.
    """

    def __init__(
        self, alpha: float = 0.1, beta: float = 1.0, eps: float = 1e-7
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob    = torch.sigmoid(logits)

        # Forward CE
        ce = F.binary_cross_entropy_with_logits(logits, targets)

        # Reverse CE: −p·log(y+ε) − (1−p)·log(1−y+ε)
        y_clamp = targets.clamp(self.eps, 1.0 - self.eps)
        rce = -(prob * torch.log(y_clamp) + (1.0 - prob) * torch.log(1.0 - y_clamp))

        return self.alpha * ce + self.beta * rce.mean()

    def __repr__(self) -> str:
        return f"SymmetricCrossEntropy(alpha={self.alpha}, beta={self.beta})"


class GeneralizedCrossEntropy(nn.Module):
    """Generalized Cross Entropy (Zhang et al. 2018).

    L_GCE = (1 − p_y^q) / q,   q ∈ (0, 1]

    where p_y = sigmoid(logit) for positive examples and
                1 − sigmoid(logit) for negative examples.

    As q → 0:  L_GCE → CE  (standard cross entropy).
    At q = 1:  L_GCE = 1 − p_y  (MAE on predicted probability).

    MAE is robust to label noise because it does not amplify the gradient
    for confident mistakes.  q ≈ 0.7 is a common default.

    Args:
        q: Interpolation parameter in (0, 1].
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {q}")
        self.q = q

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        prob    = torch.sigmoid(logits)
        p_y     = torch.where(targets == 1, prob, 1.0 - prob)
        return ((1.0 - p_y.pow(self.q)) / self.q).mean()

    def __repr__(self) -> str:
        return f"GeneralizedCrossEntropy(q={self.q})"


if __name__ == "__main__":
    import unittest

    class TestSCE(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = SymmetricCrossEntropy(alpha=0.1, beta=1.0)
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

        def test_beta_zero_equals_bce(self):
            # β=0 → pure CE = BCE
            loss_pure = SymmetricCrossEntropy(alpha=1.0, beta=0.0)
            expected  = F.binary_cross_entropy_with_logits(self.logits, self.targets)
            self.assertAlmostEqual(loss_pure(self.logits, self.targets).item(),
                                   expected.item(), places=5)

    class TestGCE(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.loss    = GeneralizedCrossEntropy(q=0.7)
            self.logits  = torch.randn(128)
            self.targets = torch.randint(0, 2, (128,)).float()

        def test_output_scalar(self):
            self.assertEqual(self.loss(self.logits, self.targets).shape, torch.Size([]))

        def test_range(self):
            # GCE ∈ [0, 1/q]
            out = self.loss(self.logits, self.targets).item()
            self.assertGreaterEqual(out, 0.0)
            self.assertLessEqual(out, 1.0 / self.loss.q + 1e-4)

        def test_backward(self):
            logits = self.logits.requires_grad_(True)
            self.loss(logits, self.targets).backward()
            self.assertIsNotNone(logits.grad)

        def test_invalid_q(self):
            with self.assertRaises(ValueError):
                GeneralizedCrossEntropy(q=0.0)
            with self.assertRaises(ValueError):
                GeneralizedCrossEntropy(q=1.5)

        def test_q_one_is_mae(self):
            # q=1 → L = (1 - p_y) = MAE on probabilities
            loss_mae = GeneralizedCrossEntropy(q=1.0)
            prob = torch.sigmoid(self.logits)
            p_y  = torch.where(self.targets == 1, prob, 1.0 - prob)
            expected = (1.0 - p_y).mean()
            self.assertAlmostEqual(loss_mae(self.logits, self.targets).item(),
                                   expected.item(), places=5)

    unittest.main()
