"""Trajectory-based traversability ground truth.

For each scan, a point is labelled traversable (1) if it lies within the
robot's 2-D footprint projected along any **future** pose — i.e. positions
the robot has not yet visited.

At init, all trajectory positions are loaded into a forward-looking set.
Each call to ``process()`` consumes (removes) the current pose so that only
what is strictly ahead of the robot is considered traversable.  This prevents
labelling obstacles behind the robot as traversable.

Output channel: ``trav_gt``  (npys — one uint8 .npy per scan)

Typical usage::

    poses = np.stack([ds[i].data["gicp_poses"] for i in range(len(ds))])
    Goose3DDataset.run_preprocess(
        TravFromTraj(poses, robot_radius=0.75, height_min=-0.3, height_max=0.5),
        split_dir,
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.spatial import KDTree

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample


class TravFromTraj(FramePreprocessor):
    """Label each point traversable if it lies in the robot's forward footprint.

    All trajectory positions are loaded at init.  At each processed frame the
    current pose is consumed, so only strictly future positions are used to
    build the traversability mask.

    Args:
        poses:          (N, 4, 4) float64 world poses (T_world_sensor),
                        one per scan.
        robot_radius:   Half-width of the robot footprint in XY (metres).
        height_min:     Minimum point height relative to the nearest robot
                        position to be traversable (metres, ≤ 0).
        height_max:     Maximum point height relative to the nearest robot
                        position (metres, ≥ 0).
        forward_window: Maximum number of future poses to look ahead.
                        ``None`` (default) uses the entire remaining trajectory.
    """

    output_key: ClassVar[str] = "trav_gt"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        poses: np.ndarray,
        robot_radius: float = 0.75,
        height_min: float = -0.3,
        height_max: float = 0.5,
        forward_window: int | None = None,
        sequence_gap: float = 5.0,
    ) -> None:
        self._poses = np.asarray(poses, dtype=np.float64)  # (N, 4, 4)
        self._robot_radius = robot_radius
        self._height_min = height_min
        self._height_max = height_max
        self._forward_window = forward_window
        self._idx = 0  # index of the current (not yet consumed) pose

        # Precompute the last-frame index of the sequence each frame belongs to.
        # A boundary is declared when consecutive pose origins are > sequence_gap m apart.
        positions = self._poses[:, :3, 3]
        dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)  # (N-1,)
        boundaries = np.where(dists > sequence_gap)[0]  # last frame index before each jump
        ends = np.concatenate([boundaries, [len(self._poses) - 1]])
        # For frame i, _seq_end[i] = index of the last frame in the same sequence
        self._seq_end = ends[np.searchsorted(ends, np.arange(len(self._poses)))]

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data["lidar"])
        xyz_sensor = pc[:, :3].astype(np.float64)
        N = len(xyz_sensor)

        # Transform scan points to world frame using current pose
        T = self._poses[self._idx]
        xyz_h = np.column_stack([xyz_sensor, np.ones(N)])
        xyz_world = (T @ xyz_h.T).T[:, :3]

        # Consume current pose: forward set starts at idx+1
        self._idx += 1
        seq_end = int(self._seq_end[self._idx - 1]) + 1  # exclusive, capped at sequence boundary
        end = min(
            self._idx + self._forward_window if self._forward_window is not None else seq_end,
            seq_end,
        )
        future_pos = self._poses[self._idx : end, :3, 3]  # (M, 3)

        if len(future_pos) == 0:
            return np.zeros(N, dtype=np.uint8)

        tree = KDTree(future_pos[:, :2])
        dist_xy, nn_idx = tree.query(xyz_world[:, :2], k=1, workers=-1)
        dz = xyz_world[:, 2] - future_pos[nn_idx, 2]

        return (
            (dist_xy < self._robot_radius)
            & (dz >= self._height_min)
            & (dz <= self._height_max)
        ).astype(np.uint8)
