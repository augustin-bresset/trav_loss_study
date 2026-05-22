from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss (Lin et al. 2017).

    Addresses class imbalance by down-weighting easy negatives.
    Input logits are raw scores (pre-sigmoid).

    Args:
        gamma: Focusing parameter — higher = more focus on hard examples.
        pos_weight: Weight for positive class (like BCE pos_weight).
                    Useful when positives are rare.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N,) raw scores.
            targets: (N,) binary labels, dtype long or float.
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=logits.device)
            if self.pos_weight is not None else None,
        )
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = bce * (1 - p_t) ** self.gamma

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
