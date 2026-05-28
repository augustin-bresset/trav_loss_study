"""Label-based traversability — maps semantic IDs to binary traversable/non-traversable.

Each point is labelled 1 if its semantic class ID is in ``traversable_ids``, 0 otherwise.

Output channel: ``trav_label``  (npys — one uint8 .npy per scan)

Default traversable IDs for RELLIS-3D::

    {1: dirt, 3: grass, 10: asphalt, 23: concrete, 31: puddle, 33: mud}

Typical usage::

    Rellis3DDataset.run_preprocess(
        TravFromLabels(),
        "/data/rellis",
    )
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample

_RELLIS_TRAVERSABLE_IDS: frozenset[int] = frozenset({1, 3, 10, 23, 31, 33})


class TravFromLabels(FramePreprocessor):
    """Label each point traversable based on its semantic class ID.

    Args:
        traversable_ids: Set of semantic class IDs considered traversable.
                         Defaults to the RELLIS-3D traversable classes.
    """

    output_key: ClassVar[str] = "trav_label"
    output_loader: ClassVar[str] = "npys"
    input_keys: ClassVar[list[str]] = ["labels"]
    timestamps_from: ClassVar[str] = "lidar"
    sources: ClassVar[list[str]] = ["labels"]

    def __init__(self, traversable_ids: frozenset[int] | None = None) -> None:
        self._trav_ids = (
            traversable_ids if traversable_ids is not None else _RELLIS_TRAVERSABLE_IDS
        )

    def process(self, sample: Sample) -> np.ndarray:
        labels = np.asarray(sample.data["labels"])
        return np.isin(labels, list(self._trav_ids)).astype(np.uint8)
