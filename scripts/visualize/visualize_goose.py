"""Interactive 3-D visualisation of GOOSE-3D traversability labels.

Launches ``apairo_visu.LidarViewer`` with a composite label that encodes:

  bit 0  (value +1)  trav_gt       — trajectory-based ground truth (green)
  bit 1  (value +2)  trav_terrain  — terrain-estimate (blue)
  bit 2  (value +4)  semantic trav — GOOSE classes mapped to traversable (purple)

Combined label in [0, 7]:
  0  none-traversable      (gray)
  1  GT only               (green)
  2  terrain only          (blue)
  3  GT + terrain          (yellow / lime)
  4  semantic only         (purple)
  5  GT + semantic         (teal)
  6  terrain + semantic    (cyan)
  7  all three agree       (white)

A trained checkpoint can optionally be overlaid as an 8th label channel
(bit 3, value +8 → labels in [0, 15]) by providing --checkpoint.

Usage:
    python -m scripts.visualize_goose --root /data/goose/GOOSE_3D --split val
    python -m scripts.visualize_goose --root /data/goose/GOOSE_3D --split val --no-semantic
    python -m scripts.visualize_goose --root /data/goose/GOOSE_3D --split val --start 42
    python -m scripts.visualize_goose --root /data/goose/GOOSE_3D --split val \\
        --checkpoint data/checkpoints/goose/bce_run/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

import apairo_visu
from apairo_visu import LidarViewer, ViewConfig
from apairo.core.sample import Sample
from src.datasets import GooseCompositeDataset


# ---------------------------------------------------------------------------
# Optional model prediction overlay
# ---------------------------------------------------------------------------


def _add_model_predictions(
    composite_ds: GooseCompositeDataset,
    checkpoint: Path,
    device: str = "cpu",
) -> "ModelOverlayDataset":
    from src.models.sparse_trav_net import SparseTravNet
    from torchsparse.utils.quantize import sparse_quantize
    from torchsparse import SparseTensor

    model = SparseTravNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return ModelOverlayDataset(composite_ds, model, device)


class ModelOverlayDataset:
    """Wrap GooseCompositeDataset to add model prediction bit (bit 3)."""

    VOXEL_SIZE = 0.1
    MAX_RAD    = 50.0

    def __init__(self, base_ds, model, device: str) -> None:
        self._base  = base_ds
        self._model = model
        self._dev   = device

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> Sample:
        from torchsparse.utils.quantize import sparse_quantize
        from torchsparse import SparseTensor

        sample  = self._base[idx]
        pc      = np.asarray(sample.data["lidar"])
        xyz     = pc[:, :3]
        inten   = pc[:, 3]
        mask    = np.linalg.norm(xyz, axis=1) < self.MAX_RAD
        xyz_f   = xyz[mask]
        inten_f = inten[mask]
        feats   = np.column_stack([xyz_f, inten_f]).astype(np.float32)

        coords_q = np.floor(xyz_f / self.VOXEL_SIZE).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)
        feats_q  = feats[sel]

        batch_coords = np.hstack([np.zeros((len(coords_q), 1), dtype=np.int32), coords_q])
        st = SparseTensor(
            coords=torch.from_numpy(batch_coords).int(),
            feats=torch.from_numpy(feats_q).float(),
        ).to(self._dev)

        with torch.no_grad():
            logits  = self._model(st)
            pred_vox = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)  # (V,)

        # Map voxel prediction back to points (via inverse quantization)
        pred_pts_masked = pred_vox[inv]  # (N_masked,)
        pred_pts = np.zeros(len(pc), dtype=np.int32)
        pred_pts[mask] = pred_pts_masked

        composite = np.asarray(sample.data["trav_composite"]).astype(np.int32)
        composite |= pred_pts << 3  # bit 3 = model prediction

        return Sample(data={
            "lidar": sample.data["lidar"],
            "trav_composite": composite,
        })


# ---------------------------------------------------------------------------
# Label config builder
# ---------------------------------------------------------------------------


def _make_label_cfg(with_model: bool = False) -> dict:
    """Build a label config dict for the composite traversability label."""
    color_map = {
        0: "#808080",  # none
        1: "#27AE60",  # GT only           — green
        2: "#2980B9",  # terrain only      — blue
        3: "#F4D03F",  # GT + terrain      — yellow
        4: "#8E44AD",  # semantic only     — purple
        5: "#1ABC9C",  # GT + semantic     — teal
        6: "#00BCD4",  # terrain + semantic — cyan
        7: "#FFFFFF",  # all three agree   — white
    }
    semantic_map = {
        0: "none",
        1: "gt_only",
        2: "terrain_only",
        3: "gt+terrain",
        4: "semantic_only",
        5: "gt+semantic",
        6: "terrain+semantic",
        7: "all_agree",
    }

    if with_model:
        extra_colors = {
            k + 8: v for k, v in color_map.items()
        }
        # tint model-predicted values with a red overlay indicator
        for k in list(extra_colors):
            r, g, b = _hex_to_rgb(extra_colors[k])
            blended = f"#{min(255, r+60):02X}{g:02X}{b:02X}"
            extra_colors[k] = blended
            semantic_map[k] = semantic_map[k - 8] + "+model"
        color_map.update(extra_colors)

    return {"color_map": color_map, "semantic_map": semantic_map}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise GOOSE traversability labels.")
    parser.add_argument("--root",       required=True, help="GOOSE_3D root (above train/val dirs).")
    parser.add_argument("--split",      default="val", choices=["train", "val"])
    parser.add_argument("--start",      type=int, default=0, help="First frame to display.")
    parser.add_argument("--no-semantic", action="store_true", help="Disable GOOSE semantic bit.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint to overlay (optional).")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    dataset = GooseCompositeDataset(
        root_dir=args.root,
        split=args.split,
        with_semantic=not args.no_semantic,
    )

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[warn] checkpoint not found: {ckpt}")
        else:
            dataset = _add_model_predictions(dataset, ckpt, args.device)

    label_cfg_path = Path(__file__).parents[1] / "resources" / "trav_composite_label_cfg.yaml"
    label_cfg = apairo_visu.load_label_config(label_cfg_path)
    view_cfg  = ViewConfig(point_key="lidar", label_key="trav_composite")

    print(f"Dataset : {args.root}  split={args.split}  ({len(dataset)} scans)")
    print("Label encoding:")
    for cid, name in sorted(label_cfg["semantic_map"].items()):
        print(f"  {cid:2d}  {name}")

    LidarViewer.launch(dataset, view_cfg=view_cfg, label_cfg=label_cfg, start_idx=args.start)


if __name__ == "__main__":
    main()
