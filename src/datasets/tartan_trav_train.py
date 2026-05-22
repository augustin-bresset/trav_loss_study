"""PyTorch Dataset for training binary traversability on TartanDrive.

Each item is one LiDAR scan voxelized into a sparse tensor.
Labels: 1 = robot drove here (positive), 0 = unlabeled.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class TartanTravTrainDataset(Dataset):
    """Per-scan dataset loading xyz + intensity + trav_label.

    Splits sequences into train/val by index (no scan-level shuffle to avoid
    temporal leakage within a sequence).

    Args:
        tartan_root: Root directory containing one sub-dir per sequence.
        voxel_size:  Quantization size in metres.
        max_rad:     Range filter (same value used during preprocessing).
        split:       'train' or 'val'.
        train_frac:  Fraction of sequences used for training.
        seed:        Seed for sequence-level shuffle before split.
    """

    def __init__(
        self,
        tartan_root: str | Path,
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.voxel_size = voxel_size
        self.max_rad = max_rad

        root = Path(tartan_root)
        seq_dirs = sorted(d for d in root.iterdir() if d.is_dir())

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(seq_dirs))
        n_train = int(len(seq_dirs) * train_frac)
        split_idx = idx[:n_train] if split == "train" else idx[n_train:]
        seq_dirs = [seq_dirs[i] for i in split_idx]

        self.items: List[Tuple[Path, Path, Path]] = []
        for seq_dir in seq_dirs:
            cloud_dir = seq_dir / "velodyne_0"
            label_dir = seq_dir / "trav_label"
            if not label_dir.exists():
                continue
            for xyz_path in sorted(cloud_dir.glob("*.npy")):
                if not xyz_path.stem.isdigit():
                    continue
                label_path = label_dir / f"{xyz_path.stem}.npy"
                intensity_path = cloud_dir / f"{xyz_path.stem}_intensity.npy"
                if label_path.exists():
                    self.items.append((xyz_path, intensity_path, label_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        xyz_path, intensity_path, label_path = self.items[idx]

        xyz = np.load(xyz_path).astype(np.float32)          # (N_raw, 3)
        labels = np.load(label_path).astype(np.int32)        # (N_filt,)
        intensity = (
            np.load(intensity_path).astype(np.float32)
            if intensity_path.exists()
            else np.ones(len(xyz), dtype=np.float32)
        )

        # Same range filter applied during preprocessing
        mask = np.linalg.norm(xyz, axis=1) < self.max_rad
        xyz = xyz[mask]
        intensity = intensity[mask]
        # labels were computed on the filtered xyz → already aligned

        feats = np.column_stack([xyz, intensity])             # (N_filt, 4)

        # Voxelization
        coords_q = np.floor(xyz / self.voxel_size).astype(np.int32)
        coords_q, sel_idx, inverse = sparse_quantize(
            coords_q,
            return_index=True,
            return_inverse=True,
        )

        feats_q = feats[sel_idx]                              # (N_vox, 4)

        # Max aggregation for labels: any positive point → positive voxel
        labels_q = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(labels_q, inverse, labels)

        return {
            "coords": torch.from_numpy(coords_q).int(),      # (N_vox, 3)
            "feats":  torch.from_numpy(feats_q).float(),     # (N_vox, 4)
            "labels": torch.from_numpy(labels_q).long(),     # (N_vox,)
        }

    @property
    def pos_ratio(self) -> float:
        """Estimated positive ratio from a sample of items (first 100)."""
        total, pos = 0, 0
        for i in range(min(100, len(self))):
            item = self[i]
            pos += item["labels"].sum().item()
            total += len(item["labels"])
        return pos / max(total, 1)


def trav_collate(batch: list) -> dict:
    """Collate items into a batched SparseTensor."""
    coords_list = [b["coords"] for b in batch]
    feats_list  = [b["feats"]  for b in batch]
    labels_list = [b["labels"] for b in batch]

    batched_coords = torch.cat([
        torch.cat([torch.full((len(c), 1), i, dtype=torch.int), c], dim=1)
        for i, c in enumerate(coords_list)
    ])
    batched_feats  = torch.cat(feats_list)
    batched_labels = torch.cat(labels_list)

    return {
        "sparse_input": SparseTensor(coords=batched_coords, feats=batched_feats),
        "labels": batched_labels,
    }
