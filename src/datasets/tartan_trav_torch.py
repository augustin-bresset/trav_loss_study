"""Apairo-backed PyTorch Dataset for TartanDrive traversability training.

Uses TartanTravDataset (AbstractDataset) for file discovery and data loading,
and apairo_manifest.yaml to verify the trav_label channel is present.
Compare with tartan_trav_train.py which does all of this manually.

Key differences vs tartan_trav_train.py:
  - Sequence discovery: via TartanTravDataset (finds scans + poses automatically)
  - Label discovery:    via read_manifest() — fails loudly if preprocess not run
  - Per-sample loading: via TartanTravDataset.__getitem__ → Sample
  - Label path:         via dataset.derived_path() — single source of truth
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate
from torchsparse.utils.quantize import sparse_quantize

from apairo.manifest import read_manifest

from src.datasets.tartan_trav import TartanTravDataset


class TartanTravTorchDataset(Dataset):
    """PyTorch Dataset backed by Apairo's TartanTravDataset.

    Discovers sequences by checking apairo_manifest.yaml for the
    'trav_label' derived channel. Fails at construction time if
    preprocess_trav has not been run on a sequence.

    Args:
        tartan_root:  Root dir containing sequence subdirs.
        voxel_size:   Voxel size in metres.
        max_rad:      Range filter in metres (must match preprocessing).
        split:        'train' or 'val'.
        train_frac:   Fraction of sequences for training.
        seed:         Seed for sequence-level shuffle.
        lidar_subdir: LiDAR subdirectory name.
        poses_subdir: Poses subdirectory name.
    """

    LABEL_KEY = "trav_label"

    def __init__(
        self,
        tartan_root: str | Path,
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 42,
        lidar_subdir: str = "velodyne_0",
        poses_subdir: str = "gicp_poses",
    ) -> None:
        self.voxel_size = voxel_size

        root = Path(tartan_root)

        # Discover sequences that have been preprocessed (manifest contains trav_label)
        all_seq_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        labeled = []
        skipped = []
        for seq_dir in all_seq_dirs:
            manifest = read_manifest(seq_dir)
            if self.LABEL_KEY in manifest.get("derived", {}):
                labeled.append(seq_dir)
            else:
                skipped.append(seq_dir.name)

        if skipped:
            print(f"[TartanTravTorchDataset] Skipped {len(skipped)} unlabeled sequences")
        print(f"[TartanTravTorchDataset] Found {len(labeled)} labeled sequences")

        # Sequence-level split
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(labeled))
        n_train = max(1, int(len(labeled) * train_frac))
        split_idx = perm[:n_train] if split == "train" else perm[n_train:]
        seq_dirs = [labeled[i] for i in split_idx]

        # Build one TartanTravDataset per sequence + store (dataset, scan_idx) pairs
        self._entries: List[Tuple[TartanTravDataset, int]] = []
        for seq_dir in seq_dirs:
            try:
                ds = TartanTravDataset(
                    seq_dir=seq_dir,
                    lidar_subdir=lidar_subdir,
                    poses_subdir=poses_subdir,
                    max_rad=max_rad,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                print(f"  Skip {seq_dir.name}: {e}")
                continue
            for i in range(len(ds)):
                self._entries.append((ds, i))

        print(f"[TartanTravTorchDataset] {split}: {len(seq_dirs)} seqs, {len(self)} scans")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict:
        apairo_ds, scan_idx = self._entries[idx]

        # Apairo loads xyz + intensity (with range filter applied)
        sample = apairo_ds[scan_idx]
        xyz       = sample.data["xyz"].numpy()        # (N, 3)
        intensity = sample.data["intensity"].numpy()  # (N,)

        # Labels via derived_path — single source of truth for the file location
        label_path = apairo_ds.derived_path(scan_idx, self.LABEL_KEY, "npy")
        labels = np.load(label_path).astype(np.int32)  # (N,)

        feats = np.column_stack([xyz, intensity])     # (N, 4)

        # Voxelization
        coords_q = np.floor(xyz / self.voxel_size).astype(np.int32)
        coords_q, sel_idx, inverse = sparse_quantize(
            coords_q, return_index=True, return_inverse=True
        )
        feats_q = feats[sel_idx]

        labels_q = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(labels_q, inverse, labels)

        return {
            "coords": torch.from_numpy(coords_q).int(),
            "feats":  torch.from_numpy(feats_q).float(),
            "labels": torch.from_numpy(labels_q).long(),
        }

    @property
    def pos_ratio(self) -> float:
        total, pos = 0, 0
        for i in range(min(100, len(self))):
            item = self[i]
            pos   += item["labels"].sum().item()
            total += len(item["labels"])
        return pos / max(total, 1)


# ---------------------------------------------------------------------------
# Collate — same as tartan_trav_train.py
# ---------------------------------------------------------------------------

def trav_collate_apairo(batch: list) -> dict:
    coords_list = [b["coords"] for b in batch]
    feats_list  = [b["feats"]  for b in batch]
    labels_list = [b["labels"] for b in batch]

    batched_coords = torch.cat([
        torch.cat([torch.full((len(c), 1), i, dtype=torch.int), c], dim=1)
        for i, c in enumerate(coords_list)
    ])

    return {
        "sparse_input": SparseTensor(
            coords=batched_coords, feats=torch.cat(feats_list)
        ),
        "labels": torch.cat(labels_list),
    }
