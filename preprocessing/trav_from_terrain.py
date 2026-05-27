"""Terrain-based traversability estimation from point-cloud geometry.

Each scan is divided into a 2-D XY grid. A cell is considered traversable
when it is locally flat (low height variance) and close to the ground plane.
The per-cell label is mapped back to per-point probabilities (float32 in
[0, 1], where 1 = traversable).

Output channel: ``trav_terrain``  (npys — one float32 .npy per scan)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class TravFromTerrain(FramePreprocessor):
    """Estimate traversability from height variance in a 2-D grid.

    Args:
        cell_size:       XY grid cell size (metres).
        max_height_var:  Maximum height variance for a cell to be traversable.
        max_height_diff: Maximum height above estimated ground for traversable cells.
        min_pts:         Minimum points per cell (cells with fewer points are
                         marked non-traversable).
    """

    output_key: ClassVar[str] = "trav_terrain"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        cell_size: float = 0.5,
        max_height_var: float = 0.04,
        max_height_diff: float = 1.5,
        min_pts: int = 3,
    ) -> None:
        self._cell_size = cell_size
        self._max_height_var = max_height_var
        self._max_height_diff = max_height_diff
        self._min_pts = min_pts

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data["lidar"])
        xyz = pc[:, :3].astype(np.float32)
        N = len(xyz)

        # 2-D grid indices
        gx = np.floor(xyz[:, 0] / self._cell_size).astype(np.int32)
        gy = np.floor(xyz[:, 1] / self._cell_size).astype(np.int32)
        gx_off = gx - gx.min()
        gy_off = gy - gy.min()
        W = int(gy.max() - gy.min()) + 1
        flat = gx_off * W + gy_off  # (N,) cell index per point

        nc = int(flat.max()) + 1
        z = xyz[:, 2].astype(np.float64)

        cnt = np.bincount(flat, minlength=nc).astype(np.float32)
        z_sum = np.bincount(flat, weights=z, minlength=nc).astype(np.float32)
        z2_sum = np.bincount(flat, weights=z ** 2, minlength=nc).astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            z_mean = np.where(cnt > 0, z_sum / cnt, 0.0)
            z_var = np.where(cnt > 0, z2_sum / cnt - z_mean ** 2, np.inf)

        # Ground plane: lowest mean height among populated cells
        populated = cnt >= self._min_pts
        ground_z = float(z_mean[populated].min()) if populated.any() else 0.0

        cell_trav = (
            populated
            & (z_var < self._max_height_var)
            & ((z_mean - ground_z) < self._max_height_diff)
        )

        return cell_trav[flat].astype(np.float32)  # (N,) ∈ {0.0, 1.0}
