"""PyTorch datasets for traversability training and visualisation.

Each class inherits from both ``torch.utils.data.Dataset`` and the relevant
apairo dataset, so it plugs into DataLoader while reusing apairo's file
discovery, profile-based loading, and split filtering.

Preprocessing must have been run via ``scripts/preprocess/`` first.

Collate
-------
``sparse_collate`` is a single generic collate function for all datasets.
It assembles per-item dicts into a batched SparseTensor under ``sparse_input``,
and concatenates ``labels`` and ``alt_labels`` (when present).

Dataset key layout (all training datasets)::

    coords       int32  (N, 3)   quantized voxel coordinates
    feats        float  (N, 4)   [x, y, z, intensity]
    labels       long   (N,)     primary GT  (trav_gt)
    alt_labels   long   (N,)     secondary metric
                                   GOOSE  → trav_terrain (terrain estimate)
                                   Rellis → trav_label   (semantic-based)

Composite (visualisation) datasets return apairo ``Sample`` objects with a
``trav_composite`` channel encoding method agreement as bit flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from apairo import Goose3DDataset, Rellis3DDataset
from apairo.core.sample import Sample


# ---------------------------------------------------------------------------
# Generic collate
# ---------------------------------------------------------------------------


def sparse_collate(batch: list) -> dict:
    """Collate training items into a batched SparseTensor.

    Works with any dataset that returns ``{coords, feats, labels, [alt_labels]}``.
    """
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
# Shared voxelisation helper
# ---------------------------------------------------------------------------


def _voxelize(
    pc: np.ndarray,
    labels: np.ndarray,
    alt: np.ndarray,
    voxel_size: float,
    max_rad: float,
) -> dict:
    xyz, intensity = pc[:, :3], pc[:, 3]

    mask = np.linalg.norm(xyz, axis=1) < max_rad
    xyz, intensity, labels, alt = xyz[mask], intensity[mask], labels[mask], alt[mask]

    feats    = np.column_stack([xyz, intensity])
    coords_q = np.floor(xyz / voxel_size).astype(np.int32)
    coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)

    labels_q = np.zeros(len(coords_q), dtype=np.int32)
    alt_q    = np.zeros(len(coords_q), dtype=np.int32)
    np.maximum.at(labels_q, inv, labels)
    np.maximum.at(alt_q,    inv, alt)

    return {
        "coords":     torch.from_numpy(coords_q).int(),
        "feats":      torch.from_numpy(feats[sel]).float(),
        "labels":     torch.from_numpy(labels_q).long(),
        "alt_labels": torch.from_numpy(alt_q).long(),
    }


# ---------------------------------------------------------------------------
# GOOSE — training
# ---------------------------------------------------------------------------


class GooseTorchDataset(Dataset, Goose3DDataset):
    """Per-scan GOOSE-3D traversability dataset for sparse-conv training.

    Inherits file discovery from ``Goose3DDataset``.
    Primary label: ``trav_gt`` (trajectory footprint).
    Secondary label: ``trav_terrain`` (terrain-height estimate).

    Args:
        root_dir:   GOOSE root directory.
        split:      ``"train"`` or ``"val"``.
        voxel_size: Voxel quantization cell size in metres.
        max_rad:    Range filter in metres.
        min_pos:    Minimum positive voxels required to keep a scan.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
        min_pos: int = 1,
    ) -> None:
        Goose3DDataset.__init__(
            self,
            root_dir=Path(root_dir),
            keys=["lidar", "trav_gt", "trav_terrain"],
            split=split,
        )
        self.voxel_size = voxel_size
        self.max_rad    = max_rad

        valid, n_skip = [], 0
        for i in range(Goose3DDataset.__len__(self)):
            if min_pos <= 0:
                valid.append(i)
                continue
            labels = Goose3DDataset.__getitem__(self, i).data["trav_gt"]
            if int((np.asarray(labels) == 1).sum()) >= min_pos:
                valid.append(i)
            else:
                n_skip += 1

        if n_skip:
            print(f"[GooseTorchDataset] {split}: skipped {n_skip} scans (< {min_pos} positive)")
        print(f"[GooseTorchDataset] {split}: {len(valid)} scans")
        self._valid = valid

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict:
        sample = Goose3DDataset.__getitem__(self, self._valid[idx])
        return _voxelize(
            pc=np.asarray(sample.data["lidar"]),
            labels=np.asarray(sample.data["trav_gt"]).astype(np.int32),
            alt=(np.asarray(sample.data["trav_terrain"]) > 0.5).astype(np.int32),
            voxel_size=self.voxel_size,
            max_rad=self.max_rad,
        )


# ---------------------------------------------------------------------------
# GOOSE — visualisation
# ---------------------------------------------------------------------------


class GooseCompositeDataset(Dataset, Goose3DDataset):
    """GOOSE-3D dataset with merged traversability label for apairo_visu.

    ``__getitem__`` returns an apairo ``Sample`` with ``trav_composite``.

    Label encoding::

        bit 0 (1): trav_gt       — trajectory GT      → green
        bit 1 (2): trav_terrain  — terrain estimate   → blue
        bit 2 (4): GOOSE semantic traversable         → (optional)

    Args:
        root_dir:          GOOSE root directory.
        split:             ``"train"`` or ``"val"``.
        terrain_threshold: Binarisation threshold for ``trav_terrain``.
        traversable_ids:   GOOSE semantic class IDs considered traversable.
        with_semantic:     Include semantic bit.
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
        self._threshold     = terrain_threshold
        self._trav_ids      = traversable_ids or self._DEFAULT_TRAV_IDS
        self._with_semantic = with_semantic

        keys = ["lidar", "trav_gt", "trav_terrain"]
        if with_semantic:
            keys.append("labels")

        Goose3DDataset.__init__(self, root_dir=Path(root_dir), keys=keys, split=split)

    def __len__(self) -> int:
        return Goose3DDataset.__len__(self)

    def __getitem__(self, idx: int) -> Sample:
        raw          = Goose3DDataset.__getitem__(self, idx)
        trav_gt      = np.asarray(raw.data["trav_gt"]).astype(np.int32)
        trav_terrain = (np.asarray(raw.data["trav_terrain"]) > self._threshold).astype(np.int32)

        combined = trav_gt | (trav_terrain << 1)

        if self._with_semantic and "labels" in raw.data:
            sem_trav  = np.isin(np.asarray(raw.data["labels"]), list(self._trav_ids)).astype(np.int32)
            combined |= sem_trav << 2

        return Sample(data={
            "lidar":          raw.data["lidar"],
            "trav_composite": combined.astype(np.int32),
        })


# ---------------------------------------------------------------------------
# Rellis — training
# ---------------------------------------------------------------------------


class RellisTorchDataset(Dataset, Rellis3DDataset):
    """Per-scan RELLIS-3D traversability dataset for sparse-conv training.

    Inherits file discovery from ``Rellis3DDataset``.
    Primary label: ``trav_gt`` (trajectory footprint).
    Secondary label: ``trav_label`` (semantic-based estimate).

    Rellis-3D has no built-in train/val split; pass ``sequences`` to select
    which numbered sequences to include (e.g. ``[0, 1, 2]`` for train,
    ``[3, 4]`` for val).

    Args:
        root_dir:   RELLIS root directory (parent of ``Rellis-3D/``).
        sequences:  Sequence indices to include.  ``None`` loads all.
        voxel_size: Voxel quantization cell size in metres.
        max_rad:    Range filter in metres.
        min_pos:    Minimum positive voxels required to keep a scan.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sequences: list[int] | None = None,
        voxel_size: float = 0.1,
        max_rad: float = 50.0,
        min_pos: int = 1,
    ) -> None:
        Rellis3DDataset.__init__(
            self,
            root_dir=Path(root_dir),
            keys=["lidar", "trav_gt", "trav_label"],
        )
        self.voxel_size = voxel_size
        self.max_rad    = max_rad

        n_total = Rellis3DDataset.__len__(self)
        valid, n_skip = [], 0
        for i in range(n_total):
            if sequences is not None:
                seq_idx = self._sequence_index(i)
                if seq_idx not in sequences:
                    continue
            if min_pos <= 0:
                valid.append(i)
                continue
            labels = Rellis3DDataset.__getitem__(self, i).data["trav_gt"]
            if int((np.asarray(labels) == 1).sum()) >= min_pos:
                valid.append(i)
            else:
                n_skip += 1

        if n_skip:
            print(f"[RellisTorchDataset]: skipped {n_skip} scans (< {min_pos} positive)")
        print(f"[RellisTorchDataset]: {len(valid)} scans")
        self._valid = valid

    def _sequence_index(self, frame_idx: int) -> int:
        """Return the sequence number for a given global frame index."""
        ref_key = self._ref_key
        path = self._files[ref_key][frame_idx]
        parts = path.relative_to(self._root).parts
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict:
        sample = Rellis3DDataset.__getitem__(self, self._valid[idx])
        return _voxelize(
            pc=np.asarray(sample.data["lidar"]),
            labels=np.asarray(sample.data["trav_gt"]).astype(np.int32),
            alt=np.asarray(sample.data["trav_label"]).astype(np.int32),
            voxel_size=self.voxel_size,
            max_rad=self.max_rad,
        )


# ---------------------------------------------------------------------------
# Rellis — visualisation
# ---------------------------------------------------------------------------


class RellisCompositeDataset(Dataset, Rellis3DDataset):
    """RELLIS-3D dataset with merged traversability label for apairo_visu.

    ``__getitem__`` returns an apairo ``Sample`` with ``trav_composite``.

    Label encoding::

        bit 0 (1): trav_gt     — trajectory GT       → green
        bit 1 (2): trav_label  — semantic-based GT   → blue
        both (3)               — full agreement       → yellow

    Args:
        root_dir: RELLIS root directory (parent of ``Rellis-3D/``).
    """

    def __init__(self, root_dir: str | Path) -> None:
        Rellis3DDataset.__init__(
            self,
            root_dir=Path(root_dir),
            keys=["lidar", "trav_gt", "trav_label"],
        )

    def __len__(self) -> int:
        return Rellis3DDataset.__len__(self)

    def __getitem__(self, idx: int) -> Sample:
        raw        = Rellis3DDataset.__getitem__(self, idx)
        trav_gt    = np.asarray(raw.data["trav_gt"]).astype(np.int32)
        trav_label = np.asarray(raw.data["trav_label"]).astype(np.int32)

        return Sample(data={
            "lidar":          raw.data["lidar"],
            "trav_composite": (trav_gt | (trav_label << 1)).astype(np.int32),
        })
