"""Incremental scan-to-scan GICP pose estimation.

Produces one (4, 4) float64 world pose per scan, accumulated via
point-to-plane ICP.  The first scan is the world origin (identity).

Cross-sequence boundaries (when the dataset mixes multiple recording
sessions) the accumulated drift will jump, but for single-session
splits the poses are consistent.

Output channel: ``gicp_poses``  (npys — one .npy per scan)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

try:
    import open3d as o3d
    _O3D_OK = True
except ImportError:
    _O3D_OK = False


class GICPPoses(FramePreprocessor):
    """Stateful scan-to-scan GICP → one (4, 4) world pose per frame.

    Uses Open3D Generalized ICP (point-to-plane).  Each call to
    :meth:`process` registers the current scan against the previous one
    and returns the accumulated world pose T_world_sensor as a (4, 4)
    float64 array.

    Args:
        voxel_size: Down-sampling voxel size (metres).
        max_rad:    Range filter applied before registration (metres).
        max_corr:   Maximum correspondence distance for ICP (metres).
    """

    output_key: ClassVar[str] = "gicp_poses"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["lidar"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["lidar"]

    def __init__(
        self,
        voxel_size: float = 0.3,
        max_rad: float = 50.0,
        max_corr: float = 1.0,
    ) -> None:
        if not _O3D_OK:
            raise ImportError("open3d is required for GICPPoses.")
        self._voxel_size = voxel_size
        self._max_rad = max_rad
        self._max_corr = max_corr
        self._prev_pcd = None
        self._T_accum = np.eye(4, dtype=np.float64)

    def process(self, sample: Sample) -> np.ndarray:
        pc = np.asarray(sample.data["lidar"])
        xyz = pc[:, :3].astype(np.float64)
        mask = np.linalg.norm(xyz, axis=1) < self._max_rad
        xyz = xyz[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd = pcd.voxel_down_sample(self._voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self._voxel_size * 3, max_nn=30)
        )

        if self._prev_pcd is not None:
            result = o3d.pipelines.registration.registration_generalized_icp(
                pcd,
                self._prev_pcd,
                self._max_corr,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            )
            # result.transformation: source → target  (current → prev)
            # T_world_cur = T_world_prev @ T_cur_to_prev
            self._T_accum = self._T_accum @ result.transformation

        self._prev_pcd = pcd
        return self._T_accum.copy()  # (4, 4) float64
