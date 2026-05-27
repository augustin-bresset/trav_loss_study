"""PyTorch Dataset for GOOSE traversability (Positive-Unlabeled learning).

Each item is one LiDAR scan voxelized into a sparse tensor.
Labels: 1 = traversable (confirmed positive), 0 = unlabeled.

Only scans with at least `min_pos` positive-labeled points are included,
so every batch is guaranteed to carry a P set for uPU / nnPU losses.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class GooseTravDataset(Dataset):
    """Per-scan dataset for GOOSE-3D traversability.

    Discovers LiDAR and trav_label files under a split root
    (e.g. ``GOOSE_3D/train/`` or ``GOOSE_3D/val/``) using the glob patterns
    ``**/lidar/**/*.bin`` and ``**/trav_label/**/*.npy``.  Both lists are
    sorted so they align by filename stem after stripping the sensor suffix.

    Args:
        root_dir:   Split root directory (GOOSE_3D/train or GOOSE_3D/val).
        voxel_size: Quantization size in metres.
        max_rad:    Range filter in metres.
        min_pos:    Minimum number of positive-labeled points required to
                    include a scan.  Scans below this threshold are skipped.
    """

    def __init__(
        self,
        root_dir: str | Path,
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
        min_pos: int = 1,
    ) -> None:
        self.voxel_size = voxel_size
        self.max_rad = max_rad

        root = Path(root_dir)
        lidar_files = sorted(root.glob("**/lidar/**/*.bin"))
        trav_files = sorted(root.glob("**/trav_label/**/*.npy"))

        if len(lidar_files) != len(trav_files):
            raise ValueError(
                f"Mismatched file counts: {len(lidar_files)} lidar vs "
                f"{len(trav_files)} trav_label in {root}"
            )

        pairs: List[Tuple[Path, Path]] = []
        n_skipped = 0
        for lidar_f, trav_f in zip(lidar_files, trav_files):
            if min_pos > 0:
                labels = np.load(trav_f)
                if int((labels == 1).sum()) < min_pos:
                    n_skipped += 1
                    continue
            pairs.append((lidar_f, trav_f))

        if n_skipped:
            print(
                f"[GooseTravDataset] {root.name}: skipped {n_skipped} scans "
                f"with < {min_pos} positive points"
            )
        print(f"[GooseTravDataset] {root.name}: {len(pairs)} scans")
        self._items = pairs

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        lidar_path, trav_path = self._items[idx]

        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        labels = np.load(trav_path).astype(np.int32)

        xyz = pc[:, :3]
        intensity = pc[:, 3]

        mask = np.linalg.norm(xyz, axis=1) < self.max_rad
        xyz = xyz[mask]
        intensity = intensity[mask]
        labels = labels[mask]

        feats = np.column_stack([xyz, intensity])  # (N, 4)

        coords_q = np.floor(xyz / self.voxel_size).astype(np.int32)
        coords_q, sel_idx, inverse = sparse_quantize(
            coords_q, return_index=True, return_inverse=True
        )
        feats_q = feats[sel_idx]

        # Any positive point in a voxel → positive voxel
        labels_q = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(labels_q, inverse, labels)

        return {
            "coords": torch.from_numpy(coords_q).int(),
            "feats":  torch.from_numpy(feats_q).float(),
            "labels": torch.from_numpy(labels_q).long(),
        }

    @property
    def pos_ratio(self) -> float:
        """Estimated positive-voxel ratio from first 100 scans."""
        total, pos = 0, 0
        for i in range(min(100, len(self))):
            item = self[i]
            pos   += item["labels"].sum().item()
            total += len(item["labels"])
        return pos / max(total, 1)


def goose_trav_collate(batch: list) -> dict:
    """Collate GooseTravDataset items into a batched SparseTensor."""
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
