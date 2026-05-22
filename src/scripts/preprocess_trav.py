"""
Generate per-point traversability labels for all TartanDrive sequences
and store them via apairo.preprocess().

Each scan gets a derived channel stored as:
    <seq_dir>/<lidar_subdir>/<stem>_trav_label.npy   (uint8, 0/1)

An apairo_manifest.yaml is written in each sequence directory to record
the derived channel so downstream datasets can discover it.

Usage:
    python -m src.scripts.preprocess_trav --config resources/tartan_preprocess.yaml

    # Dry-run (prints sequences, no writes):
    python -m src.scripts.preprocess_trav --config resources/tartan_preprocess.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Insert traversability_labeling/src/ so `from traversability.labeler import ...`
# works without conflicting with trav_loss_study's own `src/` package.
_TRAV_LABELING_SRC = Path(__file__).parents[3] / "traversability_labeling" / "src"
if str(_TRAV_LABELING_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAV_LABELING_SRC))

import apairo
from traversability.labeler import TraversabilityLabeler

from src.datasets.tartan_trav import TartanTravDataset
from src.preprocessing.trav_label_fn import TravLabelFn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get(cfg: dict, *keys, default=None):
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess TartanDrive traversability labels via apairo.preprocess."
    )
    parser.add_argument("--config", default="resources/tartan_preprocess.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sequences but do not write any files.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    root_dir     = Path(_get(cfg, "data", "source"))
    lidar_subdir = _get(cfg, "data", "lidar_subdir",  default="velodyne_0")
    poses_subdir = _get(cfg, "data", "poses_subdir",  default="gicp_poses")
    max_rad      = _get(cfg, "data", "max_rad",        default=50.0)
    robot_shape  = _get(cfg, "robot", "shape",         default="square")
    robot_size   = _get(cfg, "robot", "size",          default=3.0)
    height_min   = _get(cfg, "robot", "height_min",    default=-10.0)
    height_max   = _get(cfg, "robot", "height_max",    default=1.5)
    traj_window  = _get(cfg, "labeler", "trajectory_window", default=200)

    labeler = TraversabilityLabeler(
        robot_shape=robot_shape,
        robot_size=robot_size,
        height_min=height_min,
        height_max=height_max,
        trajectory_window=traj_window,
    )

    seq_dirs = sorted(
        p for p in root_dir.iterdir()
        if p.is_dir() and (p / lidar_subdir).is_dir()
    )
    print(f"Found {len(seq_dirs)} sequences under {root_dir}")
    if args.dry_run:
        for d in seq_dirs:
            print(f"  {d.name}")
        return

    ok, skip = 0, 0
    for seq_dir in seq_dirs:
        print(f"\n[{seq_dir.name}]", flush=True)
        try:
            dataset = TartanTravDataset(
                seq_dir=seq_dir,
                lidar_subdir=lidar_subdir,
                poses_subdir=poses_subdir,
                max_rad=max_rad,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"  Skip — {e}")
            skip += 1
            continue

        fn = TravLabelFn(labeler, dataset.poses)
        apairo.preprocess(
            dataset,
            fn,
            input_key="xyz",
            output_key="trav_label",
            output_format="npy",
        )
        print(f"  {len(dataset)} scans labeled → {dataset.cloud_dir}/", flush=True)
        ok += 1

    print(f"\nDone — {ok} sequences labeled, {skip} skipped.")


if __name__ == "__main__":
    main()
