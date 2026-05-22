"""Positive-Unlabeled (PU) learning losses for binary traversability.

In the traversability setting:
  - Positive (P): points the robot drove over  → labeled 1
  - Unlabeled (U): all other points            → labeled 0
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

    where R_U^- - π·R_P^- is an unbiased estimate of R_N^-.
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


class nnPULoss(nn.Module):
    """Non-negative PU risk estimator (Kiryo et al. 2017).

    Fixes the instability of uPU by clamping the estimated negative risk
    to be non-negative (hence nnPU):

      R(f) = π · R_P^+ + max(β, R_U^- - π · R_P^-)

    When the negative term goes below β the gradient is set to zero
    (detached from the negative branch), preventing label-flipping.

    Args:
        prior: π — estimated fraction of truly positive points among unlabeled.
        beta:  Floor for the negative risk term (default 0).
    """

    def __init__(self, prior: float = 0.1, beta: float = 0.0) -> None:
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior = prior
        self.beta = beta

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

        neg_risk = r_u_neg - self.prior * r_p_neg

        if neg_risk < self.beta:
            # Clamp: back-propagate through r_p_pos only, detach negative branch
            loss = self.prior * r_p_pos - neg_risk.detach() + self.beta
        else:
            loss = self.prior * r_p_pos + neg_risk

        return loss
