from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch

if TYPE_CHECKING:
    from traversability.labeler import TraversabilityLabeler


class TravLabelFn:
    """Stateful callable compatible with apairo.preprocess.

    apairo.preprocess iterates sequentially (idx 0, 1, …, N-1) and calls
    fn(sample.data[input_key]) without passing the index.  This class
    tracks the index internally so the correct pose is used for each scan.

    Args:
        labeler: TraversabilityLabeler instance (from traversability_labeling).
        poses:   List of N (4, 4) float64 SE3 matrices aligned with the scans.
    """

    def __init__(self, labeler: "TraversabilityLabeler", poses: List[np.ndarray]) -> None:
        self._labeler = labeler
        self._poses = poses
        self._idx = 0

    def reset(self) -> None:
        """Reset internal counter — call before re-using on a new dataset."""
        self._idx = 0

    def __call__(self, xyz_tensor: torch.Tensor) -> torch.Tensor:
        xyz = xyz_tensor.numpy().astype(np.float64)
        labels = self._labeler.label_scan(xyz, self._poses, self._idx)
        self._idx += 1
        return torch.from_numpy(labels)
