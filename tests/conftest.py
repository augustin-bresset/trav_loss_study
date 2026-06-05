"""Shared fixtures for all tests.

Synthetic data only — no real dataset files required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import _voxelize, sparse_collate
from src.models.sparse_trav_net import SparseTravNet


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def small_sparse_batch(device):
    """Two synthetic 3-D scans voxelized and batched into a SparseTensor.

    Built from random point clouds — no real dataset required.
    Labels are balanced ~50 % positive to avoid degenerate loss behaviour.
    """
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(2):
        N = 2_000
        pc = np.column_stack([
            rng.uniform(-5, 5, (N, 3)),
            rng.uniform(0, 1, N),
        ]).astype(np.float32)  # (N, 4): xyz + intensity
        labels = rng.integers(0, 2, N).astype(np.int32)
        samples.append(_voxelize(pc, labels, voxel_size=0.1, max_rad=10.0))

    batch = sparse_collate(samples)
    return batch["sparse_input"].to(device), batch["labels"].to(device)


@pytest.fixture
def model(device) -> SparseTravNet:
    """Fresh SparseTravNet for each test (function-scoped)."""
    return SparseTravNet(in_channels=4, cr=1.0).to(device)
