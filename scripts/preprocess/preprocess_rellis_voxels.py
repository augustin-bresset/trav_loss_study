"""Voxel-grid preprocessing for RELLIS-3D traversability training.

Runs three preprocessors in sequence on the whole dataset:

  1. VoxelisePointCloud  — one representative point per voxel cell.
                           Output channel: ``voxelised``
  2. VoxeliseLabels      — aggregates ``trav_gt``  (trajectory supervision).
                           Output channel: ``voxelised_trav_gt``
  3. VoxeliseLabels      — aggregates ``trav_label`` (semantic evaluation).
                           Output channel: ``voxelised_trav_label``

All three use identical ``voxel_size`` and ``max_range`` so that voxelised[i],
voxelised_trav_gt[i] and voxelised_trav_label[i] refer to the same voxel cell.

Usage:
    python -m scripts.preprocess.preprocess_rellis_voxels --config config/preprocess_rellis_voxels.yaml
    python -m scripts.preprocess.preprocess_rellis_voxels --config config/preprocess_rellis_voxels.yaml --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from apairo import Rellis3DDataset
from apairo_preprocess import VoxeliseLabels, VoxelisePointCloud

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voxel-grid preprocessing for RELLIS-3D."
    )
    parser.add_argument("--config", default="config/preprocess_rellis_voxels.yaml")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute and overwrite existing output channels.")
    args = parser.parse_args()

    cfg        = load_cfg(args.config)
    root       = Path(cfg["data"]["root"]).expanduser()
    voxel_size = float(cfg["data"]["voxel_size"])
    max_range  = float(cfg["data"]["max_rad"])
    reduction  = cfg["data"].get("reduction", "centroid")

    log.info("Root       : %s", root)
    log.info("Voxel size : %.3f m", voxel_size)
    log.info("Max range  : %.1f m", max_range)
    log.info("Reduction  : %s", reduction)

    # --- Step 1: voxelise point cloud ---
    log.info("=== Step 1/3 — VoxelisePointCloud → 'voxelised' ===")
    pc_prep = VoxelisePointCloud(
        lidar_key="lidar",
        voxel_size=voxel_size,
        max_range=max_range,
        reduction=reduction,
    )
    try:
        Rellis3DDataset.run_preprocess(pc_prep, root, overwrite=args.overwrite)
        log.info("  channel '%s' written.", pc_prep.output_key)
    except FileExistsError:
        log.info("  'voxelised' already exists — use --overwrite to recompute.")

    # --- Step 2: voxelise trav_gt (trajectory supervision) ---
    log.info("=== Step 2/3 — VoxeliseLabels(trav_gt) → 'voxelised_trav_gt' ===")
    trav_gt_prep = VoxeliseLabels(
        lidar_key="lidar",
        labels_key="trav_gt",
        voxel_size=voxel_size,
        max_range=max_range,
        aggregation="max",
        output_key="voxelised_trav_gt",
    )
    try:
        Rellis3DDataset.run_preprocess(trav_gt_prep, root, overwrite=args.overwrite)
        log.info("  channel '%s' written.", trav_gt_prep.output_key)
    except FileExistsError:
        log.info("  'voxelised_trav_gt' already exists — use --overwrite to recompute.")

    # --- Step 3: voxelise trav_label (semantic evaluation) ---
    log.info("=== Step 3/3 — VoxeliseLabels(trav_label) → 'voxelised_trav_label' ===")
    trav_label_prep = VoxeliseLabels(
        lidar_key="lidar",
        labels_key="trav_label",
        voxel_size=voxel_size,
        max_range=max_range,
        aggregation="max",
        output_key="voxelised_trav_label",
    )
    try:
        Rellis3DDataset.run_preprocess(trav_label_prep, root, overwrite=args.overwrite)
        log.info("  channel '%s' written.", trav_label_prep.output_key)
    except FileExistsError:
        log.info("  'voxelised_trav_label' already exists — use --overwrite to recompute.")

    log.info("Done.")


if __name__ == "__main__":
    main()
