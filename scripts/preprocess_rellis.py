"""Preprocess RELLIS-3D for traversability training.

Pipeline:
  1. TravFromLabels — semantic label → binary traversable  (trav_label, uint8/pt)
  2. TravFromTraj   — robot trajectory footprint → trav_gt (uint8/pt)
     Uses the native poses.txt present in each RELLIS sequence (no GICP needed).

The vault RELLIS data is read-only, so derived files are written to a local
``output_root``.  On first run, symlinks to the vault source modalities are
created inside ``output_root`` so that ``Rellis3DDataset(output_root)`` sees
both the raw channels (lidar, labels) and the derived ones (trav_label, trav_gt).

TravFromLabels is stateless → runs on the full dataset in one shot via
Rellis3DDataset.run_preprocess().

TravFromTraj is stateful (scan counter resets per sequence) → we iterate over
sequences explicitly, loading each poses.txt before running.

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
from apairo.core.sample import Sample
from preprocessing import TravFromLabels, TravFromTraj

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Source modality dirs/files to symlink into the local output root
_SOURCE_LINKS = [
    "os1_cloud_node_kitti_bin",
    "os1_cloud_node_semantickitti_label_id",
    "poses.txt",
    "calib.txt",
]


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_output_root(vault_root: Path, output_root: Path) -> list[Path]:
    """Create output_root/Rellis-3D with symlinks to vault source modalities.

    Returns the list of local sequence directories.
    """
    vault_rellis = vault_root / "Rellis-3D"
    local_rellis = output_root / "Rellis-3D"

    seq_dirs = []
    for vault_seq in sorted(p for p in vault_rellis.iterdir() if p.is_dir()):
        local_seq = local_rellis / vault_seq.name
        local_seq.mkdir(parents=True, exist_ok=True)
        for name in _SOURCE_LINKS:
            src = vault_seq / name
            dst = local_seq / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
                log.debug("  symlink %s → %s", dst, src)
        seq_dirs.append(local_seq)

    return seq_dirs


def load_rellis_poses(poses_txt: Path) -> np.ndarray:
    """Parse a RELLIS poses.txt (KITTI 3×4 row-major) → (N, 4, 4) float64."""
    rows = np.loadtxt(poses_txt, dtype=np.float64)  # (N, 12)
    n = len(rows)
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses[:, :3, :] = rows.reshape(n, 3, 4)
    poses[:, 3, 3] = 1.0
    return poses


def run_trav_labels(
    output_root: Path, trav_ids: frozenset[int], overwrite: bool
) -> None:
    log.info("=== TravFromLabels ===")
    try:
        Rellis3DDataset.run_preprocess(
            TravFromLabels(trav_ids),
            output_root,
            overwrite=overwrite,
        )
    except FileExistsError:
        log.info("  trav_label already exists — use --overwrite to recompute.")


def run_trav_traj(
    output_root: Path,
    seq_dirs: list[Path],
    cfg: dict,
    overwrite: bool,
) -> None:
    log.info("=== TravFromTraj ===")

    robot_radius   = cfg["robot"]["radius"]
    height_min     = cfg["robot"]["height_min"]
    height_max     = cfg["robot"]["height_max"]
    forward_window = cfg["labeler"].get("forward_window")  # None = full trajectory

    for seq_dir in seq_dirs:
        log.info("  Sequence: %s", seq_dir.name)

        poses_file = seq_dir / "poses.txt"
        if not poses_file.exists():
            log.warning("    poses.txt not found — skipping.")
            continue

        poses = load_rellis_poses(poses_file)
        lidar_files = sorted((seq_dir / "os1_cloud_node_kitti_bin").glob("*.bin"))

        n_frames = min(len(lidar_files), len(poses))
        if len(lidar_files) != len(poses):
            log.warning(
                "    scan/pose count mismatch (%d vs %d) — processing %d frames",
                len(lidar_files), len(poses), n_frames,
            )

        out_dir = seq_dir / "trav_gt"
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            log.info("    trav_gt already exists — use --overwrite to recompute.")
            continue
        out_dir.mkdir(exist_ok=True)

        trav_traj = TravFromTraj(
            poses=poses,
            robot_radius=robot_radius,
            height_min=height_min,
            height_max=height_max,
            forward_window=forward_window,
        )

        for i in range(n_frames):
            pc = np.fromfile(lidar_files[i], dtype=np.float32).reshape(-1, 4)
            result = trav_traj.process(Sample(data={"lidar": pc}))
            np.save(out_dir / f"{lidar_files[i].stem}.npy", result)

        log.info("    %d scans labelled → %s", n_frames, out_dir)

    Rellis3DDataset.register_channel(
        output_root,
        "trav_gt",
        "npys",
        timestamps_from="lidar",
        sources=["lidar"],
    )
    log.info("  trav_gt registered in %s/.apairo", output_root)


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

    cfg = load_cfg(args.config)
    vault_root  = Path(cfg["data"]["vault_root"])
    output_root = Path(cfg["data"]["output_root"])
    trav_ids    = frozenset(cfg.get("traversable_ids", [1, 3, 10, 23, 31, 33]))

    log.info("Vault  : %s", vault_root)
    log.info("Output : %s", output_root)

    seq_dirs = setup_output_root(vault_root, output_root)
    log.info("Found %d sequences", len(seq_dirs))

    if not args.skip_labels:
        run_trav_labels(output_root, trav_ids, args.overwrite)

    if not args.skip_traj:
        run_trav_traj(output_root, seq_dirs, cfg, args.overwrite)

    log.info("Done.")


if __name__ == "__main__":
    main()
