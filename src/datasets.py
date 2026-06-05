"""PyTorch datasets for traversability training and visualisation.

Training datasets load pre-voxelised channels from Apairo and return dicts of
tensors compatible with ``sparse_collate`` / SparseTensor.  Run
``scripts/preprocess/preprocess_rellis_voxels.py`` once before training.

Key layout::

    coords       int32  (N, 3)   quantized voxel coordinates
    feats        float  (N, 4)   [x, y, z, intensity]
    labels       long   (N,)     voxelised_trav_gt  — trajectory-based supervision
    alt_labels   long   (N,)     voxelised_trav_label — semantic-based evaluation metric

Composite datasets return apairo ``Sample`` objects for visualisation.

Use ``filter_min_pos`` to drop scans with too few positive labels before
passing a dataset to a DataLoader.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from apairo import Goose3DDataset, Rellis3DDataset
from apairo.core.sample import Sample


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def sparse_collate(batch: list) -> dict:
    batched_coords = torch.cat([
        torch.cat([torch.full((len(b["coords"]), 1), i, dtype=torch.int), b["coords"]], dim=1)
        for i, b in enumerate(batch)
    ])
    result = {
        "sparse_input": SparseTensor(
            coords=batched_coords,
            feats=torch.cat([b["feats"] for b in batch]),
        ),
        "labels": torch.cat([b["labels"] for b in batch]),
    }
    if "alt_labels" in batch[0]:
        result["alt_labels"] = torch.cat([b["alt_labels"] for b in batch])
    return result


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------


def _voxelize(
    pc: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    max_rad: float,
    alt: np.ndarray | None = None,
) -> dict:
    xyz, intensity = pc[:, :3], pc[:, 3]

    mask      = np.linalg.norm(xyz, axis=1) < max_rad
    xyz       = xyz[mask]
    intensity = intensity[mask]
    labels    = labels[mask]
    if alt is not None:
        alt = alt[mask]

    coords_q = np.floor(xyz / voxel_size).astype(np.int32)
    coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)

    labels_q = np.zeros(len(coords_q), dtype=np.int32)
    np.maximum.at(labels_q, inv, labels)

    item = {
        "coords": torch.from_numpy(coords_q).int(),
        "feats":  torch.from_numpy(np.column_stack([xyz, intensity])[sel]).float(),
        "labels": torch.from_numpy(labels_q).long(),
    }

    if alt is not None:
        alt_q = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(alt_q, inv, alt)
        item["alt_labels"] = torch.from_numpy(alt_q).long()

    return item


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _rotate_z(item: dict, voxel_size: float) -> dict:
    """Random rotation around the Z axis (gravity axis preserved).

    Rotates XY coordinates by a uniformly sampled angle, then re-quantizes
    to rebuild the voxel grid.  Voxel collisions after rotation are resolved
    by max-aggregation of labels (a positive wins over unlabeled).
    """
    theta   = torch.rand(1).item() * 2 * math.pi
    cos_t   = math.cos(theta)
    sin_t   = math.sin(theta)

    feats   = item["feats"].numpy()          # (N, 4) — [x, y, z, intensity]
    x, y, z = feats[:, 0], feats[:, 1], feats[:, 2]

    x_rot   = cos_t * x - sin_t * y
    y_rot   = sin_t * x + cos_t * y
    xyz_rot = np.column_stack([x_rot, y_rot, z])

    # Re-quantize and deduplicate
    coords_np                = np.floor(xyz_rot / voxel_size).astype(np.int32)
    coords_q, sel, inv       = sparse_quantize(coords_np, return_index=True, return_inverse=True)

    # Max-aggregate labels over merged voxels
    def _max_agg(src: torch.Tensor) -> torch.Tensor:
        out = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(out, inv, src.numpy())
        return torch.from_numpy(out).long()

    result = {
        "coords": torch.from_numpy(coords_q).int(),
        "feats":  torch.from_numpy(
            np.column_stack([xyz_rot, feats[:, 3]])[sel]
        ).float(),
        "labels": _max_agg(item["labels"]),
    }
    if "alt_labels" in item:
        result["alt_labels"] = _max_agg(item["alt_labels"])
    return result


class RotatedDataset(Dataset):
    """Wraps a cached dataset and applies random Z-rotation augmentation per sample."""

    def __init__(self, dataset: Dataset, voxel_size: float) -> None:
        self._ds         = dataset
        self._voxel_size = voxel_size

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> dict:
        return _rotate_z(self._ds[i], self._voxel_size)


# ---------------------------------------------------------------------------
# Scan filtering
# ---------------------------------------------------------------------------


def filter_min_pos(dataset, min_pos: int, label_key: str = "voxelised_trav_gt") -> Dataset:
    """Return a Subset keeping only scans with at least ``min_pos`` positive labels."""
    if min_pos <= 0:
        return dataset
    loader = dataset._loaders[label_key]
    valid  = [i for i in range(len(dataset))
              if int((np.asarray(loader[i]) == 1).sum()) >= min_pos]
    n_skip = len(dataset) - len(valid)
    if n_skip:
        print(f"[filter_min_pos] skipped {n_skip} scans (< {min_pos} positive '{label_key}')")
    return Subset(dataset, valid)


# ---------------------------------------------------------------------------
# GOOSE — training
# ---------------------------------------------------------------------------


class GooseTorchDataset(Dataset, Goose3DDataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
    ) -> None:
        Goose3DDataset.__init__(self, root_dir=Path(root_dir), keys=["lidar", "trav_gt"], split=split)
        self.voxel_size = voxel_size
        self.max_rad    = max_rad

    def __getitem__(self, idx: int) -> dict:
        sample = Goose3DDataset.__getitem__(self, idx)
        return _voxelize(
            pc=np.asarray(sample.data["lidar"]),
            labels=np.asarray(sample.data["trav_gt"]).astype(np.int32),
            voxel_size=self.voxel_size,
            max_rad=self.max_rad,
        )


# ---------------------------------------------------------------------------
# GOOSE — visualisation
# ---------------------------------------------------------------------------


class GooseCompositeDataset(Dataset, Goose3DDataset):
    """Label encoding: bit 0 = trav_gt (green), bit 1 = semantic traversable (blue)."""

    _DEFAULT_TRAV_IDS = {23, 31, 50, 51}  # asphalt, soil, low/high grass

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "val",
        traversable_ids: set[int] | None = None,
        with_semantic: bool = True,
    ) -> None:
        self._trav_ids      = traversable_ids or self._DEFAULT_TRAV_IDS
        self._with_semantic = with_semantic
        keys = ["lidar", "trav_gt"] + (["labels"] if with_semantic else [])
        Goose3DDataset.__init__(self, root_dir=Path(root_dir), keys=keys, split=split)

    def __getitem__(self, idx: int) -> Sample:
        raw      = Goose3DDataset.__getitem__(self, idx)
        combined = np.asarray(raw.data["trav_gt"]).astype(np.int32).copy()
        if self._with_semantic and "labels" in raw.data:
            combined |= np.isin(np.asarray(raw.data["labels"]), list(self._trav_ids)).astype(np.int32) << 1
        return Sample(data={"lidar": raw.data["lidar"], "trav_composite": combined})


# ---------------------------------------------------------------------------
# RELLIS — training
# ---------------------------------------------------------------------------


class RellisTorchDataset(Dataset, Rellis3DDataset):
    """Loads pre-voxelised RELLIS channels.  Run preprocess_rellis_voxels.py first.

    voxelised_trav_gt    -> labels     (trajectory-based supervision)
    voxelised_trav_label -> alt_labels (semantic-based evaluation)
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequence_ids: list[str] | None = None,
        voxel_size: float = 0.1,
    ) -> None:
        Rellis3DDataset.__init__(
            self,
            root_dir=Path(root_dir),
            keys=["voxelised", "voxelised_trav_gt", "voxelised_trav_label"],
            sequence_ids=sequence_ids,
        )
        self.voxel_size = voxel_size

    def __getitem__(self, idx: int) -> dict:
        sample = Rellis3DDataset.__getitem__(self, idx)
        pc     = np.asarray(sample.data["voxelised"], dtype=np.float32)
        coords = np.floor(pc[:, :3] / self.voxel_size).astype(np.int32)
        return {
            "coords":     torch.from_numpy(coords).int(),
            "feats":      torch.from_numpy(pc).float(),
            "labels":     torch.from_numpy(np.asarray(sample.data["voxelised_trav_gt"])).long(),
            "alt_labels": torch.from_numpy(np.asarray(sample.data["voxelised_trav_label"])).long(),
        }


# ---------------------------------------------------------------------------
# RELLIS — visualisation
# ---------------------------------------------------------------------------


class RellisCompositeDataset(Dataset, Rellis3DDataset):
    """Label encoding: bit 0 = trav_gt (green), bit 1 = trav_label (blue), both = yellow."""

    def __init__(self, root_dir: str | Path) -> None:
        Rellis3DDataset.__init__(
            self,
            root_dir=Path(root_dir),
            keys=["lidar", "trav_gt", "trav_label"],
        )

    def __getitem__(self, idx: int) -> Sample:
        raw = Rellis3DDataset.__getitem__(self, idx)
        return Sample(data={
            "lidar":          raw.data["lidar"],
            "trav_composite": (
                np.asarray(raw.data["trav_gt"]).astype(np.int32)
                | (np.asarray(raw.data["trav_label"]).astype(np.int32) << 1)
            ),
        })
