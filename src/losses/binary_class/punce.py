"""Positive-Unlabeled Contrastive Estimation (puNCE).

Acharya et al., 2022 — arXiv:2206.01206
"Positive Unlabeled Contrastive Learning"

Adapts the InfoNCE contrastive loss to the PU setting:

  L_puNCE = (1/2b) [L_P + L_U]

  L_P  — pulls labeled positives together (standard supervised InfoNCE)
  L_U  — pulls unlabeled toward positives, weighted by prior π

Note on the (1-π) self-consistency term from the paper:
  The original formulation uses augmented views z_{a(i)} to represent the
  negative component for unlabeled samples.  Without data augmentation
  (our case — single LiDAR scan per sample), this term is omitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PUNCELoss(nn.Module):
    """puNCE: Positive-Unlabeled Contrastive Estimation (Acharya et al., 2022).

    Expects L2-normalised embeddings from a projection head, NOT raw logits.
    Use :class:`SparseTravNetPUNCE` to obtain (logits, embeddings) from the
    existing backbone without modifying its weights or state dict.

    L = (L_P + L_U) / 2  (mean over the batch)

    Args:
        prior:       π — estimated fraction of positives among unlabeled.
        temperature: τ — softmax temperature (default 0.5 as in the paper).
        cls_loss:    Optional auxiliary classification loss applied to logits.
                     When provided, the forward signature accepts a tuple
                     (logits, embeddings) as first argument.
        cls_weight:  Weight λ of the classification auxiliary loss.
    """

    def __init__(
        self,
        prior: float = 0.3,
        temperature: float = 0.5,
        cls_loss: nn.Module | None = None,
        cls_weight: float = 1.0,
        max_voxels: int = 2048,
    ) -> None:
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior       = prior
        self.temperature = temperature
        self.cls_loss    = cls_loss
        self.cls_weight  = cls_weight
        self.max_voxels  = max_voxels

    def _subsample(
        self, z: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample voxels to cap similarity matrix at (max_voxels, max_voxels).

        All positives are kept; unlabeled are randomly subsampled to fill the
        budget.  Without this, z @ z.T explodes for large point clouds.
        """
        N = z.shape[0]
        if N <= self.max_voxels:
            return z, labels

        P_idx = (labels == 1).nonzero(as_tuple=True)[0]
        U_idx = (labels == 0).nonzero(as_tuple=True)[0]
        n_P   = len(P_idx)
        n_U_keep = max(0, self.max_voxels - n_P)

        if n_U_keep == 0:
            perm = torch.randperm(N, device=z.device)[:self.max_voxels]
            return z[perm], labels[perm]

        U_perm = torch.randperm(len(U_idx), device=z.device)[:n_U_keep]
        keep   = torch.cat([P_idx, U_idx[U_perm]])
        return z[keep], labels[keep]

    def _punce(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Core puNCE loss on L2-normalised embeddings.

        Args:
            z:      (N, D) L2-normalised embeddings.
            labels: (N,)   1 = labeled positive, 0 = unlabeled.
        """
        N = z.shape[0]
        P_mask = labels == 1
        U_mask = labels == 0
        n_P = P_mask.sum().item()
        n_U = U_mask.sum().item()

        if n_P < 2:
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        # Pairwise cosine similarities / τ  — shape (N, N)
        sim = (z @ z.T) / self.temperature

        # Mask diagonal with -inf so it contributes 0 to log_softmax denominator
        eye = torch.eye(N, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))

        # log-softmax over each row: log P(k | i) for k ≠ i  — (N, N)
        log_prob = F.log_softmax(sim, dim=1)

        # Replace diagonal -inf with 0 before any multiplication to avoid -inf*0=NaN
        log_prob = log_prob.masked_fill(eye, 0.0)

        # ── L_P : pull labeled positives together ───────────────────────────
        # For each positive i: average log P(j|i) over all j ∈ P \ {i}
        P_cross = P_mask.unsqueeze(0) & P_mask.unsqueeze(1)   # (N, N)
        P_cross.fill_diagonal_(False)                          # exclude self
        loss_P = -(log_prob * P_cross).sum(dim=1)[P_mask] / (n_P - 1)
        loss_P = loss_P.mean()

        # ── L_U : pull unlabeled toward positives weighted by π ─────────────
        # For each unlabeled i: π * mean_{j∈P} log P(j|i)
        # (1-π) augmented-view term omitted — no augmentation available
        if n_U == 0:
            loss_U = torch.zeros(1, device=z.device).squeeze()
        else:
            U_P_log = log_prob[U_mask][:, P_mask]   # (n_U, n_P)
            loss_U  = -self.prior * U_P_log.mean()

        return (loss_P + loss_U) / 2

    def forward(
        self,
        model_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model_output: either a (N,) logit tensor (puNCE only, no cls_loss)
                          or a tuple (logits (N,), embeddings (N, D)).
            labels:       (N,) binary — 1 = positive, 0 = unlabeled.
        """
        if isinstance(model_output, tuple):
            logits, z = model_output
        else:
            logits, z = model_output, None

        if z is None:
            raise ValueError(
                "PUNCELoss requires L2-normalised embeddings. "
                "Use SparseTravNetPUNCE which returns (logits, embeddings)."
            )

        z = F.normalize(z.float(), dim=-1)
        z, labels_sub = self._subsample(z, labels)
        loss = self._punce(z, labels_sub)

        if self.cls_loss is not None:
            loss = loss + self.cls_weight * self.cls_loss(logits, labels.float())

        return loss
