"""PyTorch datasets for GOOSE-3D traversability training and visualisation.

Backed by ``apairo.Goose3DDataset``.  Preprocessing (GICPPoses + TravFromTraj +
TravFromTerrain) must have been run via ``scripts/preprocess_goose.py`` first.

Two classes are provided:

``GooseTravTorchDataset``
    Standard training/validation dataset. Voxelises each scan into a sparse
    tensor and returns trajectory-based GT labels (``trav_gt``).

``GooseTravCompositeDataset``
    Thin wrapper for visualisation.  Merges ``trav_gt`` and ``trav_terrain``
    (and optionally GOOSE semantic labels) into a single integer label per
    point for display in ``apairo_visu``.

    Combined label encoding::

        0  non-traversable by any method  (gray)
        1  GT-only traversable            (green)
        2  terrain-only traversable       (blue)
        3  traversable by both methods    (yellow)

    When semantic labels are included, GOOSE semantic traversability is encoded
    in bit 2, giving values 0-7.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from apairo import Goose3DDataset
from apairo.core.sample import Sample


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------


class GooseTravTorchDataset(Dataset):
    """Per-scan GOOSE traversability dataset for sparse-conv training.

    Args:
        root_dir:   GOOSE split root (e.g. ``GOOSE_3D/train``).
        split:      ``"train"`` or ``"val"`` — passed to Goose3DDataset.
        voxel_size: Quantization cell size in metres.
        max_rad:    Range filter in metres.
        min_pos:    Minimum number of positive voxels required to include a scan.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
        min_pos: int = 1,
    ) -> None:
        self.voxel_size = voxel_size
        self.max_rad = max_rad

        self._ds = Goose3DDataset(
            Path(root_dir),
            keys=["lidar", "trav_gt", "trav_terrain"],
            split=split,
        )

        valid: List[int] = []
        n_skip = 0
        for i in range(len(self._ds)):
            if min_pos <= 0:
                valid.append(i)
                continue
            labels = self._ds[i].data["trav_gt"]
            if int((np.asarray(labels) == 1).sum()) >= min_pos:
                valid.append(i)
            else:
                n_skip += 1

        if n_skip:
            print(f"[GooseTravTorchDataset] {split}: skipped {n_skip} scans (< {min_pos} positive)")
        print(f"[GooseTravTorchDataset] {split}: {len(valid)} scans")
        self._valid = valid

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict:
        sample = self._ds[self._valid[idx]]
        pc = np.asarray(sample.data["lidar"])
        labels = np.asarray(sample.data["trav_gt"]).astype(np.int32)
        terrain = (np.asarray(sample.data["trav_terrain"]) > 0.5).astype(np.int32)

        xyz = pc[:, :3]
        intensity = pc[:, 3]

        mask = np.linalg.norm(xyz, axis=1) < self.max_rad
        xyz, intensity, labels, terrain = xyz[mask], intensity[mask], labels[mask], terrain[mask]

        feats = np.column_stack([xyz, intensity])  # (N, 4)

        coords_q = np.floor(xyz / self.voxel_size).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)
        feats_q = feats[sel]

        # Any positive point in a voxel → positive voxel
        labels_q = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(labels_q, inv, labels)

        terrain_q = np.zeros(len(coords_q), dtype=np.int32)
        np.maximum.at(terrain_q, inv, terrain)

        return {
            "coords":          torch.from_numpy(coords_q).int(),
            "feats":           torch.from_numpy(feats_q).float(),
            "labels":          torch.from_numpy(labels_q).long(),
            "terrain_labels":  torch.from_numpy(terrain_q).long(),
        }


def goose_trav_collate(batch: list) -> dict:
    """Collate GooseTravTorchDataset items into a batched SparseTensor."""
    batched_coords = torch.cat([
        torch.cat([torch.full((len(b["coords"]), 1), i, dtype=torch.int), b["coords"]], dim=1)
        for i, b in enumerate(batch)
    ])
    return {
        "sparse_input": SparseTensor(
            coords=batched_coords,
            feats=torch.cat([b["feats"] for b in batch]),
        ),
        "labels":         torch.cat([b["labels"]         for b in batch]),
        "terrain_labels": torch.cat([b["terrain_labels"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Composite visualisation dataset
# ---------------------------------------------------------------------------


class GooseTravCompositeDataset:
    """Wrap Goose3DDataset to expose a merged traversability label for apairo_visu.

    Combines ``trav_gt`` (trajectory GT), ``trav_terrain`` (terrain estimate),
    and optionally GOOSE semantic ``labels`` into a single integer label:

    * bit 0 (value 1): GT traversable (trav_gt == 1)
    * bit 1 (value 2): terrain traversable (trav_terrain > threshold)
    * bit 2 (value 4): GOOSE semantic traversable (label in traversable_ids)

    The resulting label lies in [0, 7].  Use the ``trav_composite.yaml``
    label config for colour mapping.

    Args:
        root_dir:          GOOSE root (above train/val dirs).
        split:             ``"train"`` or ``"val"``.
        terrain_threshold: Threshold on ``trav_terrain`` float to binarise it.
        traversable_ids:   GOOSE semantic class IDs considered traversable.
                           Defaults to the IDs in ``goose_cfg_trav.yaml``.
        with_semantic:     Include GOOSE semantic bit.
    """

    _DEFAULT_TRAV_IDS = {23, 31, 50, 51}  # asphalt, soil, low/high grass

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "val",
        terrain_threshold: float = 0.5,
        traversable_ids: set[int] | None = None,
        with_semantic: bool = True,
    ) -> None:
        self._threshold = terrain_threshold
        self._trav_ids = traversable_ids if traversable_ids is not None else self._DEFAULT_TRAV_IDS
        self._with_semantic = with_semantic

        keys = ["lidar", "trav_gt", "trav_terrain"]
        if with_semantic:
            keys.append("labels")

        self._ds = Goose3DDataset(Path(root_dir), keys=keys, split=split)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Sample:
        raw = self._ds[idx]
        trav_gt = np.asarray(raw.data["trav_gt"]).astype(np.uint8)
        trav_terrain = (np.asarray(raw.data["trav_terrain"]) > self._threshold).astype(np.uint8)

        combined = trav_gt.astype(np.int32) | (trav_terrain.astype(np.int32) << 1)

        if self._with_semantic and "labels" in raw.data:
            sem = np.asarray(raw.data["labels"])
            sem_trav = np.isin(sem, list(self._trav_ids)).astype(np.int32)
            combined |= sem_trav << 2

        return Sample(data={
            "lidar": raw.data["lidar"],
            "trav_composite": combined.astype(np.int32),
        })
