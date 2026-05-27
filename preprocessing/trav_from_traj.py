"""Trajectory-based traversability ground truth.

For each scan, a point is labelled traversable (1) if it lies within the
robot's 2-D footprint projected along any pose in a sliding temporal window
and within the configured height band relative to the robot.

This preprocessor is stateful: it must be called in scan order (0, 1, …, N-1)
and requires the full pose sequence to be loaded in advance.

Output channel: ``trav_gt``  (npys — one uint8 .npy per scan)

Typical usage::

    poses_ds = Goose3DDataset(split_dir, keys=["gicp_poses"])
    poses = np.stack([poses_ds[i].data["gicp_poses"] for i in range(len(poses_ds))])

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
    """Label each point traversable if it lies in the robot's swept footprint.

    Args:
        poses:             (N, 4, 4) float64 world poses (T_world_sensor),
                           one per scan — typically the output of GICPPoses.
        robot_radius:      Half-width of the robot footprint in XY (metres).
        height_min:        Minimum height of a point relative to the robot
                           centre to be considered traversable (metres, ≤0).
        height_max:        Maximum height relative to robot centre (metres, ≥0).
        trajectory_window: Number of past and future scans used to build the
                           trajectory window for each scan.
    """

    output_key: ClassVar[str] = "trav_gt"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar", "gicp_poses"]

    def __init__(
        self,
        poses: np.ndarray,
        robot_radius: float = 0.75,
        height_min: float = -0.3,
        height_max: float = 0.5,
        trajectory_window: int = 50,
    ) -> None:
        self._poses = np.asarray(poses, dtype=np.float64)  # (N, 4, 4)
        self._robot_radius = robot_radius
        self._height_min = height_min
        self._height_max = height_max
        self._traj_window = trajectory_window
        self._idx = 0

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data["lidar"])
        xyz_sensor = pc[:, :3].astype(np.float64)
        N = len(xyz_sensor)

        # Transform scan points to world frame
        T = self._poses[self._idx]
        xyz_h = np.column_stack([xyz_sensor, np.ones(N)])  # (N, 4)
        xyz_world = (T @ xyz_h.T).T[:, :3]  # (N, 3)

        # Sliding window of trajectory poses
        i_start = max(0, self._idx - self._traj_window)
        i_end = min(len(self._poses), self._idx + self._traj_window + 1)
        traj_pos = self._poses[i_start:i_end, :3, 3]  # (W, 3) — world-frame origins

        # For each point: distance to nearest trajectory position (XY only)
        tree = KDTree(traj_pos[:, :2])
        dist_xy, nn_idx = tree.query(xyz_world[:, :2], k=1, workers=-1)

        # Height relative to the nearest robot position
        dz = xyz_world[:, 2] - traj_pos[nn_idx, 2]

        trav = (
            (dist_xy < self._robot_radius)
            & (dz >= self._height_min)
            & (dz <= self._height_max)
        )

        self._idx += 1
        return trav.astype(np.uint8)
