"""Preprocess a GOOSE-3D split for traversability training.

Pipeline (per split):
  1. GICPPoses       — scan-to-scan GICP → gicp_poses channel (4×4 per scan)
  2. TravFromTraj    — trajectory footprint → trav_gt channel (uint8 per point)
  3. TravFromTerrain — height-variance grid → trav_terrain channel (float32 per point)

Each step is registered in .apairo so the dataset can load derived channels.

Usage:
    python -m scripts.preprocess_goose --config resources/goose_preprocess.yaml
    python -m scripts.preprocess_goose --config resources/goose_preprocess.yaml --split val
    python -m scripts.preprocess_goose --config resources/goose_preprocess.yaml --overwrite
    python -m scripts.preprocess_goose --config resources/goose_preprocess.yaml --skip-gicp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))

from apairo import Goose3DDataset
from apairo_preprocess import GICPPoses, TravFromTraj, TravFromTerrain

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_split(split_dir: Path, cfg: dict, *, overwrite: bool, skip_gicp: bool) -> None:
    log.info("=== Split: %s ===", split_dir)

    # ------------------------------------------------------------------ GICP
    if not skip_gicp:
        log.info("Step 1/3  GICPPoses")
        gicp = GICPPoses(
            voxel_size=cfg["gicp"]["voxel_size"],
            max_rad=cfg["gicp"]["max_rad"],
            max_corr=cfg["gicp"]["max_corr"],
        )
        try:
            Goose3DDataset.run_preprocess(gicp, split_dir, overwrite=overwrite)
        except FileExistsError:
            log.info("  gicp_poses already exists — use --overwrite to recompute.")
    else:
        log.info("Step 1/3  GICPPoses — skipped (--skip-gicp)")

    # --------------------------------------------------------- Load all poses
    log.info("Loading gicp_poses …")
    ds_poses = Goose3DDataset(split_dir, keys=["gicp_poses"])
    all_poses = np.stack(
        [np.asarray(ds_poses[i].data["gicp_poses"]) for i in range(len(ds_poses))]
    )  # (N, 4, 4)
    log.info("  %d poses loaded", len(all_poses))

    # -------------------------------------------------------- TravFromTraj
    log.info("Step 2/3  TravFromTraj")
    trav_traj = TravFromTraj(
        poses=all_poses,
        robot_radius=cfg["robot"]["radius"],
        height_min=cfg["robot"]["height_min"],
        height_max=cfg["robot"]["height_max"],
        forward_window=cfg["labeler"].get("forward_window"),
    )
    try:
        Goose3DDataset.run_preprocess(trav_traj, split_dir, overwrite=overwrite)
    except FileExistsError:
        log.info("  trav_gt already exists — use --overwrite to recompute.")

    # ---------------------------------------------------- TravFromTerrain
    log.info("Step 3/3  TravFromTerrain")
    trav_terrain = TravFromTerrain(
        cell_size=cfg["terrain"]["cell_size"],
        max_height_var=cfg["terrain"]["max_height_var"],
        max_height_diff=cfg["terrain"]["max_height_diff"],
        min_pts=cfg["terrain"]["min_pts"],
    )
    try:
        Goose3DDataset.run_preprocess(trav_terrain, split_dir, overwrite=overwrite)
    except FileExistsError:
        log.info("  trav_terrain already exists — use --overwrite to recompute.")

    log.info("Done: %s", split_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess GOOSE-3D for traversability.")
    parser.add_argument("--config", default="resources/goose_preprocess.yaml")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-gicp", action="store_true", help="Skip GICP if poses already exist.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    goose_root = Path(cfg["data"]["root"])

    splits = []
    if args.split in ("train", "both"):
        splits.append(goose_root / "train")
    if args.split in ("val", "both"):
        splits.append(goose_root / "val")

    for split_dir in splits:
        if not split_dir.is_dir():
            log.warning("Split dir not found: %s — skipping.", split_dir)
            continue
        run_split(split_dir, cfg, overwrite=args.overwrite, skip_gicp=args.skip_gicp)


if __name__ == "__main__":
    main()
