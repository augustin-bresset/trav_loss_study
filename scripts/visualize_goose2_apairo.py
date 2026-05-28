"""Interactive multi-viewport traversability visualisation on GOOSE-3D.

Opens one viewport per selected model checkpoint (+ one Ground Truth viewport),
all synchronized on the same frame. Model inference runs in a background thread
per viewport via apairo_visu's Pipeline system.

Label encoding (composite bits):
  bit 0 (+1)   trav_gt       — trajectory ground truth     green
  bit 1 (+2)   trav_terrain  — terrain estimate             blue
  bit 2 (+4)   semantic_trav — GOOSE semantic classes       purple
  bit 3 (+8)   model_pred    — trained model prediction     orange tint

Keyboard shortcuts (from apairo_visu):
  → / L        next frame
  ← / H        previous frame
  T            cycle colour mode (semantic / intensity / height)
  B            bird's-eye view
  R            reset camera

Panel — "Active pipelines" section:
  Each model has a checkbox. Unchecking hides its viewport immediately and
  the remaining viewports redistribute the available width. Re-checking
  triggers inference on the current frame before the viewport reappears.

Usage:
    # Ground-truth only (no model):
    python -m scripts.visualize_goose2

    # One model by experiment name:
    python -m scripts.visualize_goose2 --checkpoints nnpu_prior30

    # Compare several models side by side:
    python -m scripts.visualize_goose2 --checkpoints nnpu_prior30 focal_g3_pw6 bce

    # Full path also accepted:
    python -m scripts.visualize_goose2 --checkpoints data/checkpoints/goose/nnpu_prior30/best.pth

    # All options:
    python -m scripts.visualize_goose2 \\
        --root /mnt/vault-fellowship/goose/GOOSE_3D \\
        --split val \\
        --checkpoints nnpu_prior30 nnpu_prior40 \\
        --threshold 0.5 \\
        --device cuda \\
        --no-semantic \\
        --start 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── project + apairo_visu on path ────────────────────────────────────────────
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "apairo_visu"))

import apairo_visu
from apairo_visu import LidarViewer, ViewConfig, Pipeline
from datasets.goose_trav import GooseTravCompositeDataset
from src.losses import TRAV_LOSSES          # noqa: F401 — ensures losses registered
from src.models.sparse_trav_net import SparseTravNet

# ── constants ─────────────────────────────────────────────────────────────────
CKPT_BASE   = _ROOT / "data" / "checkpoints" / "goose"
LABEL_CFG   = _ROOT / "resources" / "trav_composite_label_cfg.yaml"
VOXEL_SIZE  = 0.1   # must match training
MAX_RAD     = 50.0  # must match training


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _list_checkpoints() -> list[Path]:
    return sorted(CKPT_BASE.glob("*/best.pth"))


def _resolve_checkpoint(spec: str) -> Path:
    """Accept an experiment name or a direct path."""
    p = Path(spec)
    if p.exists():
        return p
    candidate = CKPT_BASE / spec / "best.pth"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Checkpoint not found: '{spec}'\n"
        f"  tried: {p}\n"
        f"  tried: {candidate}\n"
        f"  available: {[c.parent.name for c in _list_checkpoints()]}"
    )


def _pick_checkpoints_interactively() -> list[Path]:
    available = _list_checkpoints()
    if not available:
        print(f"No checkpoints found under {CKPT_BASE}")
        return []

    print("\nAvailable checkpoints:")
    for i, p in enumerate(available):
        print(f"  [{i:2d}]  {p.parent.name}")
    print()
    raw = input(
        "Enter indices or names to load (space-separated), or press Enter for GT only: "
    ).strip()
    if not raw:
        return []

    selected = []
    for token in raw.split():
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < len(available):
                selected.append(available[idx])
            else:
                print(f"  [warn] index {idx} out of range, skipped")
        else:
            try:
                selected.append(_resolve_checkpoint(token))
            except FileNotFoundError as e:
                print(f"  [warn] {e}")
    return selected


# ---------------------------------------------------------------------------
# Model inference pipeline step
# ---------------------------------------------------------------------------

def _load_model(ckpt: Path, device: str) -> SparseTravNet:
    model = SparseTravNet(in_channels=4, cr=1.0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def make_inference_step(model: SparseTravNet, device: str):
    """Return a Pipeline step that ORs the model prediction into bit 3 of labels."""
    from torchsparse import SparseTensor
    from torchsparse.utils.quantize import sparse_quantize

    def _inference(pts: np.ndarray, labels: np.ndarray | None):
        """
        pts:    (N, 4)  x y z intensity
        labels: (N,)    composite int32 (bits 0-2 already set)
        """
        xyz       = pts[:, :3].astype(np.float32)
        intensity = pts[:, 3].astype(np.float32)

        # Range filter (inference only — does not drop points from the display)
        mask = np.linalg.norm(xyz, axis=1) < MAX_RAD
        xyz_f   = xyz[mask]
        inten_f = intensity[mask]

        if len(xyz_f) == 0:
            return pts, labels

        # Voxelise
        coords_q = np.floor(xyz_f / VOXEL_SIZE).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(
            coords_q, return_index=True, return_inverse=True
        )
        feats_q = np.column_stack([xyz_f[sel], inten_f[sel]])

        batch_coords = np.hstack(
            [np.zeros((len(coords_q), 1), dtype=np.int32), coords_q]
        )
        st = SparseTensor(
            coords=torch.from_numpy(batch_coords).int(),
            feats=torch.from_numpy(feats_q).float(),
        ).to(device)

        with torch.no_grad():
            logits   = model(st)
            pred_vox = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)

        # Voxel → point mapping (points outside max_rad stay 0)
        pred_pts = np.zeros(len(pts), dtype=np.int32)
        pred_pts[mask] = pred_vox[inv]

        out_labels = (labels if labels is not None else np.zeros(len(pts), dtype=np.int32))
        return pts, out_labels | (pred_pts << 3)  # set bit 3

    return _inference


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GOOSE-3D traversability evaluation viewer.")
    parser.add_argument(
        "--root", default="/mnt/vault-fellowship/goose/GOOSE_3D",
        help="GOOSE_3D root directory (contains train/ and val/).",
    )
    parser.add_argument("--split",       default="val",  choices=["train", "val"])
    parser.add_argument("--checkpoints", nargs="*",      metavar="CKPT",
                        help="Experiment names or paths. Omit for interactive selection.")
    parser.add_argument("--threshold",   type=float,     default=0.5,
                        help="Binarisation threshold for trav_terrain (default 0.5).")
    parser.add_argument("--device",      default="cpu",  help="Inference device (cpu / cuda).")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable GOOSE semantic bit in composite label.")
    parser.add_argument("--start",       type=int,       default=0,
                        help="Index of the first frame to display.")
    args = parser.parse_args()

    # ── resolve checkpoints ──────────────────────────────────────────────────
    if args.checkpoints is None:
        ckpt_paths = _pick_checkpoints_interactively()
    else:
        ckpt_paths = [_resolve_checkpoint(s) for s in args.checkpoints]

    # ── dataset ──────────────────────────────────────────────────────────────
    print(f"\nLoading dataset  {args.root}  split={args.split} …")
    dataset = GooseTravCompositeDataset(
        root_dir=args.root,
        split=args.split,
        terrain_threshold=args.threshold,
        with_semantic=not args.no_semantic,
    )
    print(f"  {len(dataset)} scans")

    # ── build pipelines ──────────────────────────────────────────────────────
    pipelines: list[Pipeline] = [
        Pipeline("Ground Truth"),  # no steps — shows raw composite bits 0-2
    ]

    for ckpt in ckpt_paths:
        exp_name = ckpt.parent.name
        print(f"  Loading model: {exp_name} …")
        model = _load_model(ckpt, args.device)
        step  = make_inference_step(model, args.device)
        pipelines.append(Pipeline(f"Model: {exp_name}", [step]))

    # ── viewer config ────────────────────────────────────────────────────────
    view_cfg  = ViewConfig(
        point_key="lidar",
        label_key="trav_composite",
        intensity_channel=3,
    )
    label_cfg = apairo_visu.load_label_config(LABEL_CFG)

    # ── print legend ─────────────────────────────────────────────────────────
    print("\nLabel legend:")
    for cid, name in sorted(label_cfg["semantic_map"].items()):
        hex_col = label_cfg["color_map"].get(cid, "#808080")
        print(f"  {cid:2d}  {hex_col}  {name}")

    print(f"\nViewports ({len(pipelines)}):")
    for p in pipelines:
        print(f"  · {p.name}")

    print("\nControls: ← → (or H/L) navigate  |  T colour mode  |  B bird's-eye  |  R reset\n")

    # ── launch ───────────────────────────────────────────────────────────────
    LidarViewer.launch(
        dataset,
        view_cfg=view_cfg,
        label_cfg=label_cfg,
        pipelines=pipelines,
        start_idx=args.start,
    )


if __name__ == "__main__":
    main()
