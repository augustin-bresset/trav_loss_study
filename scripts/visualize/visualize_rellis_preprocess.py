"""Visualise the RELLIS preprocessing pipeline output for verification.

Reads the derived channels (trav_label, trav_gt) produced by
``scripts/preprocess_rellis.py`` and displays them side-by-side in three
apairo_visu viewports.

Viewports:
  · Combined      — trav_gt (bit 0, green) + trav_label (bit 1, blue) overlaid
  · Traj GT       — TravFromTraj output only (green / gray)
  · Sem Label     — TravFromLabels output only (blue / gray)

Composite encoding (values 0--3):
  bit 0 (+1) : trav_gt    — trajectory ground truth  →  green
  bit 1 (+2) : trav_label — semantic-based label     →  blue
  both (3)               — full agreement            →  yellow

Usage:
    python -m scripts.visualize.visualize_rellis_preprocess
    python -m scripts.visualize.visualize_rellis_preprocess --config resources/rellis_preprocess.yaml
    python -m scripts.visualize.visualize_rellis_preprocess --start 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import apairo_visu
from apairo_visu import LidarViewer, ViewConfig, Pipeline
from src.datasets import RellisCompositeDataset

LABEL_CFG = _ROOT / "resources" / "trav_composite_label_cfg.yaml"


def _step_gt_only(pts: np.ndarray, labels: np.ndarray | None):
    return pts, (labels & 1) if labels is not None else labels


def _step_label_only(pts: np.ndarray, labels: np.ndarray | None):
    return pts, (labels & 2) if labels is not None else labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify RELLIS preprocessing output visually."
    )
    parser.add_argument("--config", default="resources/rellis_preprocess.yaml")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    cfg_path = (
        Path(args.config) if Path(args.config).is_absolute()
        else _ROOT / args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data"]["root"])
    print(f"Loading from: {root}")

    dataset = RellisCompositeDataset(root)
    print(f"  {len(dataset)} scans")

    pipelines = [
        Pipeline("Combined",               steps=[]),
        Pipeline("Traj GT (TravFromTraj)", steps=[_step_gt_only]),
        Pipeline("Sem (TravFromLabels)",   steps=[_step_label_only]),
    ]

    label_cfg = apairo_visu.load_label_config(str(LABEL_CFG))
    view_cfg  = ViewConfig(
        point_key="lidar",
        label_key="trav_composite",
        intensity_channel=3,
    )

    print("\nColour legend:")
    print("  gray   — non-traversable")
    print("  green  — trav_gt only  (trajectory GT)")
    print("  blue   — trav_label only  (semantic-based)")
    print("  yellow — both agree")
    print("\nControls: ← → (or H/L) navigate  |  T colour mode  |  B bird's-eye  |  R reset\n")

    LidarViewer.launch(
        dataset,
        view_cfg=view_cfg,
        label_cfg=label_cfg,
        pipelines=pipelines,
        start_idx=args.start,
    )


if __name__ == "__main__":
    main()
