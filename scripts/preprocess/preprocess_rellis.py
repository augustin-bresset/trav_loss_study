"""Preprocess RELLIS-3D for traversability training.

Pipeline:
  1. TravFromLabels — semantic label → binary traversable  (trav_label, uint8/pt)
  2. TravFromTraj   — robot trajectory footprint → trav_gt (uint8/pt)

Both steps delegate to ``Rellis3DDataset.run_preprocess()``, which handles file
placement via ``derived_path()``, format-specific writing, and channel
registration in ``.apairo``.

Usage:
    python scripts/preprocess_rellis.py --config resources/rellis_preprocess.yaml
    python scripts/preprocess_rellis.py --config resources/rellis_preprocess.yaml --overwrite
    python scripts/preprocess_rellis.py --config resources/rellis_preprocess.yaml --skip-traj
    python scripts/preprocess_rellis.py --config resources/rellis_preprocess.yaml --skip-labels
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from apairo import Rellis3DDataset
from apairo_preprocess import TravFromLabels, TravFromTraj

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _3x4_to_4x4(T34: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = T34
    return T


def run_trav_labels(root: Path, trav_ids: frozenset[int], overwrite: bool) -> None:
    log.info("=== TravFromLabels ===")
    try:
        Rellis3DDataset.run_preprocess(
            TravFromLabels(trav_ids),
            root,
            overwrite=overwrite,
        )
    except FileExistsError:
        log.info("  trav_label already exists — use --overwrite to recompute.")


def run_trav_traj(root: Path, cfg: dict, overwrite: bool) -> None:
    log.info("=== TravFromTraj ===")

    ds = Rellis3DDataset(root, keys=["poses"])
    poses = np.stack([_3x4_to_4x4(ds[i].data["poses"]) for i in range(len(ds))])
    log.info("  %d poses loaded", len(poses))

    try:
        Rellis3DDataset.run_preprocess(
            TravFromTraj(
                poses=poses,
                robot_radius=cfg["robot"]["radius"],
                height_min=cfg["robot"]["height_min"],
                height_max=cfg["robot"]["height_max"],
                forward_window=cfg["labeler"].get("forward_window"),
            ),
            root,
            overwrite=overwrite,
        )
    except FileExistsError:
        log.info("  trav_gt already exists — use --overwrite to recompute.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess RELLIS-3D traversability labels."
    )
    parser.add_argument("--config", default="resources/rellis_preprocess.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip TravFromLabels (trav_label).")
    parser.add_argument("--skip-traj", action="store_true",
                        help="Skip TravFromTraj (trav_gt).")
    args = parser.parse_args()

    cfg  = load_cfg(args.config)
    root = Path(cfg["data"]["root"])
    trav_ids = frozenset(cfg.get("traversable_ids", [1, 3, 10, 23, 31, 33]))

    log.info("Root: %s", root)

    if not args.skip_labels:
        run_trav_labels(root, trav_ids, args.overwrite)

    if not args.skip_traj:
        run_trav_traj(root, cfg, args.overwrite)

    log.info("Done.")


if __name__ == "__main__":
    main()
