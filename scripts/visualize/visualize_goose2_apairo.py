"""Compare multiple trained models on GOOSE-3D with Rerun.

One viewport per model checkpoint + one Ground Truth viewport, all
synchronized on the same timeline.

GT viewport   : composite label (trav_gt | semantic, values 0-3)
Model viewport: binary model prediction (0 = not trav, 1 = trav)

Usage:
    # Ground-truth only:
    python -m scripts.visualize.visualize_goose2_apairo --root /data/goose/GOOSE_3D

    # Compare models:
    python -m scripts.visualize.visualize_goose2_apairo \\
        --root /data/goose/GOOSE_3D \\
        --checkpoints data/checkpoints/goose/bce_run/best.pth \\
                      data/checkpoints/goose/focal_run/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import apairo_rr
from apairo_rr import Pipeline
from src.datasets import GooseCompositeDataset

CKPT_BASE  = _ROOT / "data" / "checkpoints" / "goose"
VOXEL_SIZE = 0.1
MAX_RAD    = 50.0

TRAV_COMPOSITE_CFG = {
    "color_map": {
        0: [128, 128, 128],
        1: [39,  174,  96],
        2: [41,  128, 185],
        3: [244, 208,  63],
    },
    "semantic_map": {
        0: "non-traversable",
        1: "trav-gt only",
        2: "semantic only",
        3: "gt + semantic",
    },
}

MODEL_CFG = {
    "color_map":    {0: [200, 60, 60], 1: [50, 200, 80]},
    "semantic_map": {0: "not traversable", 1: "traversable"},
}


def _resolve_checkpoint(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    candidate = CKPT_BASE / spec / "best.pth"
    if candidate.exists():
        return candidate
    available = sorted(CKPT_BASE.glob("*/best.pth"))
    raise FileNotFoundError(
        f"Checkpoint not found: '{spec}'\n"
        f"  available: {[c.parent.name for c in available]}"
    )


def make_inference_step(checkpoint: Path, device: str):
    from src.models.sparse_trav_net import SparseTravNet
    from torchsparse import SparseTensor
    from torchsparse.utils.quantize import sparse_quantize

    model = SparseTravNet(in_channels=4, cr=1.0).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    def _infer(pts: np.ndarray, labels: np.ndarray | None):
        xyz, intensity = pts[:, :3].astype(np.float32), pts[:, 3].astype(np.float32)
        mask = np.linalg.norm(xyz, axis=1) < MAX_RAD
        xyz_f, inten_f = xyz[mask], intensity[mask]
        if len(xyz_f) == 0:
            return pts, np.zeros(len(pts), dtype=np.int64)

        coords_q = np.floor(xyz_f / VOXEL_SIZE).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)
        feats = np.column_stack([xyz_f[sel], inten_f[sel]])
        bc    = np.hstack([np.zeros((len(coords_q), 1), dtype=np.int32), coords_q])

        st = SparseTensor(
            coords=torch.from_numpy(bc).int(),
            feats=torch.from_numpy(feats).float(),
        ).to(device)
        with torch.no_grad():
            pred_vox = (torch.sigmoid(model(st)) > 0.5).cpu().numpy().astype(np.int64)

        pred_pts = np.zeros(len(pts), dtype=np.int64)
        pred_pts[mask] = pred_vox[inv]
        return pts, pred_pts

    return _infer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",        required=True)
    parser.add_argument("--split",       default="val", choices=["train", "val"])
    parser.add_argument("--checkpoints", nargs="*", metavar="CKPT",
                        help="Experiment names or .pth paths.")
    parser.add_argument("--start",       type=int, default=0)
    parser.add_argument("--every",       type=int, default=1)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    dataset = GooseCompositeDataset(
        root_dir=args.root,
        split=args.split,
        with_semantic=not args.no_semantic,
    )
    print(f"Dataset: {len(dataset)} scans  ({args.split})")

    pipelines  = [Pipeline("GT composite")]
    label_cfgs = [TRAV_COMPOSITE_CFG]

    for spec in (args.checkpoints or []):
        try:
            ckpt = _resolve_checkpoint(spec)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            continue
        name = ckpt.parent.name
        print(f"  Loading checkpoint: {name}")
        pipelines.append(Pipeline(f"Model: {name}",
                                  [make_inference_step(ckpt, args.device)]))
        label_cfgs.append(MODEL_CFG)

    frames = range(args.start, len(dataset), args.every)

    apairo_rr.view(
        dataset,
        label_cfgs=label_cfgs,
        label_key="trav_composite",
        point_key="lidar",
        pipelines=pipelines,
        frames=frames,
        application_id="goose_model_comparison",
    )


if __name__ == "__main__":
    main()
