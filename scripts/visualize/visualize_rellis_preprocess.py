"""Visualise RELLIS-3D preprocessing output with Rerun.

Three synchronized viewports:
  Combined      — trav_gt (bit 0) ⊕ trav_label (bit 1)
  Traj GT       — TraversabilityFromTrajectory output only
  Sem Labels    — TraversabilityFromLabels output only

Usage:
    python -m scripts.visualize.visualize_rellis_preprocess --config resources/rellis_preprocess.yaml
    python -m scripts.visualize.visualize_rellis_preprocess --start 10 --every 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import apairo_rr
from apairo_rr import Pipeline
from src.datasets import RellisCompositeDataset

COMBINED_CFG = {
    "color_map": {
        0: [128, 128, 128],
        1: [39,  174,  96],
        2: [41,  128, 185],
        3: [244, 208,  63],
    },
    "semantic_map": {
        0: "non-traversable",
        1: "traj-GT only",
        2: "label only",
        3: "both agree",
    },
}

TRAJ_CFG = {
    "color_map":    {0: [128, 128, 128], 1: [39, 174, 96]},
    "semantic_map": {0: "non-traversable", 1: "traversable (traj-GT)"},
}

LABEL_CFG = {
    "color_map":    {0: [128, 128, 128], 1: [41, 128, 185]},
    "semantic_map": {0: "non-traversable", 1: "traversable (sem-labels)"},
}


def _gt_only(pts: np.ndarray, labels):
    return pts, (labels & 1) if labels is not None else labels


def _label_only(pts: np.ndarray, labels):
    return pts, ((labels >> 1) & 1) if labels is not None else labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="resources/rellis_preprocess.yaml")
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--every",  type=int, default=1)
    args = parser.parse_args()

    cfg_path = (
        Path(args.config) if Path(args.config).is_absolute()
        else _ROOT / args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root    = Path(cfg["data"]["root"])
    dataset = RellisCompositeDataset(root)
    print(f"Loading from: {root}  ({len(dataset)} scans)")

    apairo_rr.view(
        dataset,
        label_cfgs=[COMBINED_CFG, TRAJ_CFG, LABEL_CFG],
        label_key="trav_composite",
        point_key="lidar",
        pipelines=[
            Pipeline("Combined"),
            Pipeline("Traj GT",    [_gt_only]),
            Pipeline("Sem Labels", [_label_only]),
        ],
        frames=range(args.start, len(dataset), args.every),
        application_id="rellis_preprocessing",
    )


if __name__ == "__main__":
    main()
