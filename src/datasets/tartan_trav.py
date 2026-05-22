from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from apairo.core.abstract_dataset import AbstractDataset
from apairo.core.sample import Sample


class TartanTravDataset(AbstractDataset):
    """Apairo AbstractDataset for one TartanDrive sequence.

    Loads velodyne scans (xyz + intensity) and GICP poses.
    Implements derived_path() so apairo.preprocess() stores
    per-point derived channels (e.g. trav_label) alongside the scans:
        <seq_dir>/<lidar_subdir>/<stem>_<key>.<ext>
    """

    synchronous: bool = True

    def __init__(
        self,
        seq_dir: str | Path,
        lidar_subdir: str = "velodyne_0",
        poses_subdir: str = "gicp_poses",
        max_rad: float = 50.0,
    ) -> None:
        self.seq_dir = Path(seq_dir)
        self.cloud_dir = self.seq_dir / lidar_subdir
        self.root_dir = self.seq_dir
        self.max_rad = max_rad

        self.cloud_files: List[Path] = sorted(
            p for p in self.cloud_dir.glob("*.npy")
            if p.stem.isdigit()
        )
        if not self.cloud_files:
            raise FileNotFoundError(f"No scan files found in {self.cloud_dir}")

        self.poses: List[np.ndarray] = self._load_poses(self.seq_dir / poses_subdir)

        if len(self.poses) != len(self.cloud_files):
            raise ValueError(
                f"Pose count ({len(self.poses)}) != scan count ({len(self.cloud_files)}) "
                f"in {self.seq_dir.name}"
            )

        self._set_keys(["xyz", "intensity"])

    # ------------------------------------------------------------------

    def _load_poses(self, poses_dir: Path) -> List[np.ndarray]:
        raw = np.load(poses_dir / "poses.npy")         # (N, 4, 4)
        valid = np.load(poses_dir / "valid_mask.npy")  # (N,) bool
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            raise RuntimeError(f"No valid poses in {poses_dir}")
        poses = []
        for i in range(len(raw)):
            if valid[i]:
                poses.append(raw[i])
            else:
                nearest = valid_idx[np.abs(valid_idx - i).argmin()]
                poses.append(raw[nearest])
        return poses

    # ------------------------------------------------------------------
    # AbstractDataset interface

    def __len__(self) -> int:
        return len(self.cloud_files)

    def __getitem__(self, idx: int) -> Sample:
        path = self.cloud_files[idx]
        xyz = np.load(path).astype(np.float32)

        intensity_path = path.parent / (path.stem + "_intensity.npy")
        intensity = (
            np.load(intensity_path).astype(np.float32)
            if intensity_path.exists()
            else np.ones(len(xyz), dtype=np.float32)
        )

        mask = np.linalg.norm(xyz, axis=1) < self.max_rad
        return Sample(data={
            "xyz":       torch.from_numpy(xyz[mask]),
            "intensity": torch.from_numpy(intensity[mask]),
        })

    def derived_path(self, idx: int, key: str, ext: str) -> Path:
        stem = self.cloud_files[idx].stem
        return self.seq_dir / key / f"{stem}.{ext}"

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self) -> Sample:
        if self._iter_idx >= len(self):
            raise StopIteration
        sample = self[self._iter_idx]
        self._iter_idx += 1
        return sample

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.seq_dir.name
