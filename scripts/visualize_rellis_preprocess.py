"""Visualise the RELLIS preprocessing pipeline output for verification.

Reads the derived channels (trav_label, trav_gt) produced by
``scripts/preprocess_rellis.py`` via ``Rellis3DDataset`` and displays them
side-by-side in three apairo_visu viewports.

Viewports:
  · Combined      — trav_gt (bit 0, green) + trav_label (bit 1, blue) overlaid
  · Traj GT       — TravFromTraj output only (green / gray)
  · Sem Label     — TravFromLabels output only (blue / gray)

Composite encoding (reuses trav_composite_label_cfg.yaml, values 0--3):
  bit 0 (+1) : trav_gt    — trajectory ground truth  →  green
  bit 1 (+2) : trav_label — semantic-based label     →  blue
  both (3)               — full agreement            →  yellow

Usage:
    python -m scripts.visualize_rellis_preprocess
    python -m scripts.visualize_rellis_preprocess --config resources/rellis_preprocess.yaml
    python -m scripts.visualize_rellis_preprocess --start 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "apairo_visu"))

import apairo
import apairo_visu
from apairo_visu import LidarViewer, ViewConfig, Pipeline
from apairo.core.sample import Sample

LABEL_CFG = _ROOT / "resources" / "trav_composite_label_cfg.yaml"


# ---------------------------------------------------------------------------
# Thin composite wrapper (mirrors GooseTravCompositeDataset)
# ---------------------------------------------------------------------------

class RellisTravCompositeDataset:
    """Wraps Rellis3DDataset to expose a merged trav label for apairo_visu.

    Combines trav_gt (trajectory GT, bit 0) and trav_label (semantic-based,
    bit 1) into a single integer label per point.  Values 0--3.

    Preprocessing must have been run first (preprocess_rellis.py).
    """

    def __init__(self, output_root: str | Path) -> None:
        self._ds = apairo.Rellis3DDataset(
            Path(output_root),
            keys=["lidar", "trav_label", "trav_gt"],
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Sample:
        raw = self._ds[idx]
        trav_gt    = np.asarray(raw.data["trav_gt"]).astype(np.int32)
        trav_label = np.asarray(raw.data["trav_label"]).astype(np.int32)
        composite  = trav_gt | (trav_label << 1)
        return Sample(data={
            "lidar":          raw.data["lidar"],
            "trav_composite": composite,
        })


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_gt_only(pts: np.ndarray, labels: np.ndarray | None):
    """Keep only bit 0 (trav_gt → green)."""
    return pts, (labels & 1) if labels is not None else labels


def _step_label_only(pts: np.ndarray, labels: np.ndarray | None):
    """Keep only bit 1 (trav_label → blue, stays as value 2)."""
    return pts, (labels & 2) if labels is not None else labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify RELLIS preprocessing output visually."
    )
    parser.add_argument(
        "--config", default="resources/rellis_preprocess.yaml",
        help="Path to rellis_preprocess.yaml.",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting frame index (default: 0).",
    )
    args = parser.parse_args()

    cfg_path = (
        Path(args.config) if Path(args.config).is_absolute()
        else _ROOT / args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data"]["root"])
    print(f"Loading from: {root}")

    dataset = RellisTravCompositeDataset(root)
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
