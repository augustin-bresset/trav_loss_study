"""Tests for SparseTravNet and the torchsparse backend.

Three concerns:
  1. torchsparse installation — CUDA kernels must be present
  2. Model initialisation — correct output shape, no NaN at init
  3. Model learning capacity — gradients flow, loss decreases on a fixed batch
"""

import torch
import torch.nn.functional as F
import pytest

import torchsparse.backend as ts_backend

from src.models.sparse_trav_net import SparseTravNet


# ---------------------------------------------------------------------------
# 1. torchsparse backend
# ---------------------------------------------------------------------------

def test_torchsparse_cuda_backend_available():
    """Regression: CPU-only torchsparse wheel causes silent NaN during training.

    The subm kernel is required for every sparse conv forward pass.
    If this test fails, rebuild torchsparse from source with CUDA support:
        CUDA_HOME=... pip install git+https://github.com/mit-han-lab/torchsparse.git
    """
    assert hasattr(ts_backend, "build_kernel_map_subm_hashmap"), (
        "torchsparse CPU-only build detected — CUDA kernels are missing. "
        "All forward passes will raise AttributeError (silently swallowed by "
        "the trainer), producing loss=NaN and metrics=0 for all 75 runs."
    )


def test_torchsparse_cuda_backend_downsample():
    """Downsampling kernel (used in encoder strides) must also be present."""
    assert hasattr(ts_backend, "build_kernel_map_downsample_hashmap")


# ---------------------------------------------------------------------------
# 2. Model initialisation
# ---------------------------------------------------------------------------

def test_forward_output_shape(model, small_sparse_batch):
    """One logit per voxel — shape must match the label tensor."""
    st, labels = small_sparse_batch
    with torch.no_grad():
        logits = model(st)
    assert logits.shape == labels.shape, (
        f"Expected {labels.shape}, got {logits.shape}"
    )


def test_forward_no_nan_at_init(model, small_sparse_batch):
    """Randomly initialised model must not produce NaN or Inf."""
    st, _ = small_sparse_batch
    with torch.no_grad():
        logits = model(st)
    assert not torch.isnan(logits).any(),  "NaN in model output at initialisation"
    assert torch.isfinite(logits).all(),   "Inf in model output at initialisation"


def test_forward_output_is_unbounded(model, small_sparse_batch):
    """Head has no activation — logits should not be clipped to [0, 1]."""
    st, _ = small_sparse_batch
    with torch.no_grad():
        logits = model(st)
    # A random model will almost certainly produce values outside [0, 1]
    assert (logits.abs() > 0.01).any(), "Logits look suspiciously small — check the head"


# ---------------------------------------------------------------------------
# 3. Gradient flow and learning capacity
# ---------------------------------------------------------------------------

def test_all_parameters_receive_gradients(model, small_sparse_batch):
    """Every named parameter must get a gradient after one backward pass."""
    st, labels = small_sparse_batch
    model.train()
    logits = model(st)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    loss.backward()

    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"


def test_no_nan_gradients(model, small_sparse_batch):
    """Gradients must be finite after one backward pass."""
    st, labels = small_sparse_batch
    model.train()
    logits = model(st)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    loss.backward()

    nan_params = [
        name for name, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not nan_params, f"NaN gradients in: {nan_params}"


def test_model_overfits_single_batch(small_sparse_batch, device):
    """Model must be able to memorize one batch — basic learning capacity check.

    Uses a fresh model and 40 Adam steps on the same batch.
    Final loss must be < 90 % of initial loss.
    """
    model = SparseTravNet(in_channels=4, cr=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    st, labels = small_sparse_batch

    model.train()
    initial_loss = None
    for step in range(40):
        optimizer.zero_grad()
        logits = model(st)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.9, (
        f"Loss did not decrease by ≥10 %: {initial_loss:.4f} → {final_loss:.4f}"
    )
