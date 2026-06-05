"""Tests for all binary-class losses in src/losses/binary_class/.

Each loss is exercised with the same configs used in cv_rellis.py.
Three properties are checked per loss:
  - finite scalar output on a mixed (pos + neg) batch
  - finite gradients on the logits
  - loss is strictly positive (sanity check on the formula)
"""

import pytest
import torch

from src.losses.binary_class import TRAV_LOSSES


# Configs mirror EXPERIMENTS in scripts/train/cv_rellis.py
LOSS_CONFIGS = [
    ("bce",            {"name": "bce"}),
    ("focal",          {"name": "focal",         "gamma": 2.0, "pos_weight": 3.6}),
    ("asl",            {"name": "asl",            "gamma_neg": 4.0, "asl_clip": 0.05}),
    ("tversky",        {"name": "tversky",        "tversky_alpha": 0.3, "tversky_beta": 0.7}),
    ("focal_tversky",  {"name": "focal_tversky",  "tversky_alpha": 0.3, "tversky_beta": 0.7}),
    ("lovasz",         {"name": "lovasz"}),
    ("bce_dice",       {"name": "bce_dice"}),
    ("bce_lovasz",     {"name": "bce_lovasz"}),
    ("focal_dice",     {"name": "focal_dice",     "gamma": 2.0}),
    ("sce",            {"name": "sce",            "sce_alpha": 0.1, "sce_beta": 1.0}),
    ("gce",            {"name": "gce",            "gce_q": 0.7}),
    ("nnpu_p20",       {"name": "bce_nnpu",       "prior": 0.20}),
    ("nnpu_p30",       {"name": "bce_nnpu",       "prior": 0.30}),
    ("nnpu_p40",       {"name": "bce_nnpu",       "prior": 0.40}),
    ("focal_nnpu_p30", {"name": "focal_nnpu",     "prior": 0.30, "gamma": 2.0}),
]

IDS = [name for name, _ in LOSS_CONFIGS]


@pytest.fixture(scope="module")
def mixed_batch(device):
    """Fixed logits and ~50 % positive labels — reused across all loss tests."""
    torch.manual_seed(0)
    N = 512
    logits  = torch.randn(N, device=device)
    labels  = torch.randint(0, 2, (N,), device=device).float()
    return logits, labels


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_returns_finite_scalar(name, cfg, mixed_batch, device):
    """Loss must be a finite scalar on a normal mixed batch."""
    logits, labels = mixed_batch
    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)

    assert loss.ndim == 0,            f"{name}: expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss).all(), f"{name}: loss is not finite: {loss.item()}"


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_is_positive(name, cfg, mixed_batch, device):
    """All losses are non-negative by construction."""
    logits, labels = mixed_batch
    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    assert loss.item() >= 0.0, f"{name}: negative loss {loss.item():.6f}"


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_gradient_is_finite(name, cfg, mixed_batch, device):
    """Gradients on logits must be finite after backward."""
    logits_base, labels = mixed_batch
    logits = logits_base.detach().requires_grad_(True)

    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    loss.backward()

    assert logits.grad is not None, f"{name}: no gradient"
    assert torch.isfinite(logits.grad).all(), (
        f"{name}: NaN/Inf gradient — "
        f"max_abs={logits.grad.abs().max().item():.4f}"
    )


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_all_positive_labels(name, cfg, device):
    """Loss must remain finite when all labels are 1 (common in early training)."""
    torch.manual_seed(1)
    logits = torch.randn(256, device=device)
    labels = torch.ones(256, device=device)

    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    assert torch.isfinite(loss), f"{name}: not finite on all-positive batch: {loss.item()}"


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_all_negative_labels(name, cfg, device):
    """Loss must remain finite when all labels are 0."""
    torch.manual_seed(2)
    logits = torch.randn(256, device=device)
    labels = torch.zeros(256, device=device)

    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    assert torch.isfinite(loss), f"{name}: not finite on all-negative batch: {loss.item()}"


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_finite_with_extreme_logits(name, cfg, device):
    """Loss must be finite with saturating logits (as from a freshly-init model)."""
    torch.manual_seed(3)
    logits = torch.empty(512, device=device).uniform_(-50.0, 50.0)
    labels = torch.randint(0, 2, (512,), device=device).float()

    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    assert torch.isfinite(loss), f"{name}: not finite with extreme logits: {loss.item()}"


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_gradient_finite_with_extreme_logits(name, cfg, device):
    """Gradients must be finite with saturating logits — catches 0^0 NaN in focusing terms."""
    torch.manual_seed(4)
    logits = torch.empty(512, device=device).uniform_(-50.0, 50.0).requires_grad_(True)
    labels = torch.randint(0, 2, (512,), device=device).float()

    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), (
        f"{name}: NaN/Inf gradient with extreme logits — "
        f"max_abs={logits.grad.abs().max().item():.4f}"
    )


@pytest.mark.parametrize("name,cfg", LOSS_CONFIGS, ids=IDS)
def test_loss_finite_with_imbalanced_batch(name, cfg, device):
    """Loss must be finite on a highly imbalanced batch (~5% positive) like RELLIS."""
    torch.manual_seed(5)
    N = 512
    logits = torch.randn(N, device=device)
    labels = (torch.rand(N, device=device) < 0.05).float()

    criterion = TRAV_LOSSES[cfg["name"]](cfg).to(device)
    loss = criterion(logits, labels)
    assert torch.isfinite(loss), f"{name}: not finite on imbalanced batch: {loss.item()}"
