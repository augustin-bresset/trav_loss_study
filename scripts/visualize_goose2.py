"""Standalone GOOSE-3D traversability visualiser (no apairo / apairo_visu).

Reads LiDAR (.bin), semantic labels (.label) and trav_label (.npy) directly
from the GOOSE filesystem.  Loads model checkpoints on demand.

Display modes (Combobox):
  GT traversability   — trav_label GT (green / gray)
  Semantic trav.      — GOOSE semantic classes flagged as traversable (purple/gray)
  Composite           — GT (bit 0) | semantic (bit 1) combined, 4-class colour map
  Intensity           — grayscale by LiDAR intensity
  Model: <name>       — model prediction (orange / gray)    [added when model loaded]
  Model: <name> vs GT — TP / FP / FN / TN comparison       [added when model loaded]

Model checkboxes on the right panel enable lazy loading and add the model's
display modes to the combobox.  Unchecking removes them.

Navigation: ← / → arrow keys  or  H / L  or  Prev / Next buttons.

Usage:
    python scripts/visualize_goose2.py
    python scripts/visualize_goose2.py --checkpoints nnpu_prior30 focal_g3_pw6
    python scripts/visualize_goose2.py --root /mnt/vault-fellowship/goose/GOOSE_3D \\
        --split val --device cpu --start 0
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.sparse_trav_net import SparseTravNet

# ── constants ─────────────────────────────────────────────────────────────────
GOOSE_ROOT   = "/mnt/vault-fellowship/goose/GOOSE_3D"
CKPT_BASE    = _ROOT / "data" / "checkpoints" / "goose"
VOXEL_SIZE   = 0.1
MAX_RAD      = 50.0
TRAV_SEM_IDS = frozenset({23, 24, 31, 50, 51})  # GOOSE classes labelled traversable

_FRAME_RE = re.compile(r"^(.+?)(?:_pcl|_goose)$")

# ── colour palette (RGB floats 0-1) ───────────────────────────────────────────
_C = {
    "gt_pos":   np.array([0.153, 0.682, 0.376]),  # green
    "sem_pos":  np.array([0.557, 0.267, 0.678]),  # purple
    "both":     np.array([0.957, 0.816, 0.247]),  # yellow
    "model":    np.array([0.902, 0.494, 0.133]),  # orange
    "tp":       np.array([0.153, 0.682, 0.376]),  # green
    "fp":       np.array([0.161, 0.502, 0.725]),  # blue
    "fn":       np.array([0.769, 0.118, 0.227]),  # red
    "neg":      np.array([0.420, 0.420, 0.420]),  # gray
}

# ── composite colour map (same as trav_composite_label_cfg.yaml) ──────────────
_HEX = {
    0: "#808080", 1: "#27AE60", 2: "#8E44AD", 3: "#F4D03F",
    4: "#E67E22", 5: "#58D68D", 6: "#BB8FCE", 7: "#F9E79F",
}
_CMAP: dict[int, np.ndarray] = {
    k: np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]) / 255.0
    for k, h in _HEX.items()
}
# bit 0 → GT, bit 1 → semantic, bit 2 → model  (values 0-7)


# ─────────────────────────────────────────────────────────────────────────────
# Frame index
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    lidar_path: Path
    sem_path:   Path | None
    trav_path:  Path | None
    frame_id:   str

    def load(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return (pts (N,4) float32, sem_labels (N,) int32 | None, trav_gt (N,) int32 | None)."""
        pts  = np.fromfile(self.lidar_path, dtype=np.float32).reshape(-1, 4)
        sem  = np.fromfile(self.sem_path,   dtype=np.int32)               if self.sem_path  else None
        trav = np.load(self.trav_path).astype(np.int32)                   if self.trav_path else None
        return pts, sem, trav


def build_frame_index(root: Path, split: str) -> list[Frame]:
    split_root = root / split

    def _index(paths: list[Path]) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for p in sorted(paths):
            m = _FRAME_RE.match(p.stem)
            if m:
                out[m.group(1)] = p
        return out

    lidar = _index(list(split_root.rglob("*_pcl.bin")))
    sem   = _index(list(split_root.rglob("*_goose.label")))
    trav  = _index(list(split_root.rglob("*_goose.npy")))

    return [
        Frame(lp, sem.get(fid), trav.get(fid), fid)
        for fid, lp in sorted(lidar.items())
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

class ModelPredictor:
    def __init__(self, ckpt: Path, device: str = "cpu") -> None:
        self.name   = ckpt.parent.name
        self.device = device
        net = SparseTravNet(in_channels=4, cr=1.0).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        net.eval()
        self._net = net

    def predict(self, pts: np.ndarray) -> np.ndarray:
        """Return binary (N,) int32 prediction for the given point cloud."""
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        xyz   = pts[:, :3].astype(np.float32)
        inten = pts[:, 3].astype(np.float32)
        mask  = np.linalg.norm(xyz, axis=1) < MAX_RAD
        xyz_f, inten_f = xyz[mask], inten[mask]

        if len(xyz_f) == 0:
            return np.zeros(len(pts), dtype=np.int32)

        coords_q = np.floor(xyz_f / VOXEL_SIZE).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)
        feats_q = np.column_stack([xyz_f[sel], inten_f[sel]])
        batch_c = np.hstack([np.zeros((len(coords_q), 1), dtype=np.int32), coords_q])

        st = SparseTensor(
            coords=torch.from_numpy(batch_c).int(),
            feats=torch.from_numpy(feats_q).float(),
        ).to(self.device)

        with torch.no_grad():
            pred_vox = (torch.sigmoid(self._net(st)) > 0.5).cpu().numpy().astype(np.int32)

        pred = np.zeros(len(pts), dtype=np.int32)
        pred[mask] = pred_vox[inv]
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# Colourisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _color_gt(trav: np.ndarray) -> np.ndarray:
    c = np.tile(_C["neg"], (len(trav), 1))
    c[trav == 1] = _C["gt_pos"]
    return c


def _color_semantic(sem: np.ndarray) -> np.ndarray:
    c = np.tile(_C["neg"], (len(sem), 1))
    c[np.isin(sem, list(TRAV_SEM_IDS))] = _C["sem_pos"]
    return c


def _color_composite(trav: np.ndarray | None, sem: np.ndarray | None) -> np.ndarray:
    n = len(trav) if trav is not None else len(sem)
    bits = np.zeros(n, dtype=np.int32)
    if trav is not None:
        bits |= (trav == 1).astype(np.int32)           # bit 0
    if sem is not None:
        bits |= np.isin(sem, list(TRAV_SEM_IDS)).astype(np.int32) << 1  # bit 1
    return np.array([_CMAP.get(int(b), _C["neg"]) for b in bits])


def _color_model(pred: np.ndarray) -> np.ndarray:
    c = np.tile(_C["neg"], (len(pred), 1))
    c[pred == 1] = _C["model"]
    return c


def _color_vs_gt(pred: np.ndarray, trav: np.ndarray) -> np.ndarray:
    c = np.tile(_C["neg"], (len(pred), 1))  # TN = gray
    c[(trav == 1) & (pred == 1)] = _C["tp"]
    c[(trav == 0) & (pred == 1)] = _C["fp"]
    c[(trav == 1) & (pred == 0)] = _C["fn"]
    return c


def _color_intensity(pts: np.ndarray) -> np.ndarray:
    v = pts[:, 3]
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    return np.column_stack([v, v, v])


# ─────────────────────────────────────────────────────────────────────────────
# Viewer
# ─────────────────────────────────────────────────────────────────────────────

_BASE_MODES = [
    "GT traversability",
    "Semantic trav.",
    "Composite (GT + Sem)",
    "Intensity",
]

_LEGEND = [
    ("GT traversable",       "#27AE60"),
    ("Semantic traversable", "#8E44AD"),
    ("Both (GT + Sem)",      "#F4D03F"),
    ("Model prediction",     "#E67E22"),
    ("TP (GT & model)",      "#27AE60"),
    ("FP (model only)",      "#295BA8"),
    ("FN (GT only)",         "#C41E3A"),
    ("Non-traversable",      "#6B6B6B"),
]

_PANEL_W = 280


class TravViewer:
    _GEOM = "pcd"

    def __init__(self, frames: list[Frame], predictors: list[ModelPredictor]) -> None:
        self._frames     = frames
        self._idx        = 0
        self._predictors = {p.name: p for p in predictors}
        self._enabled:   dict[str, bool] = {name: False for name in self._predictors}
        # cache: idx → {"pts", "sem", "trav", "preds": {name: arr}}
        self._cache: dict[int, dict] = {}

        app = gui.Application.instance
        app.initialize()
        self._build_ui()
        self._app = app

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._win = gui.Application.instance.create_window(
            "GOOSE Traversability Viewer", 1440, 900
        )
        self._win.set_on_layout(self._on_layout)
        self._win.set_on_close(lambda: True)
        self._win.set_on_key(self._on_key)

        # 3-D scene
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self._win.renderer)
        self._win.add_child(self._scene)

        mat = rendering.MaterialRecord()
        mat.shader     = "defaultUnlit"
        mat.point_size = 2.0
        self._mat = mat

        # Right panel
        panel = gui.ScrollableVert(4, gui.Margins(8, 8, 8, 8))
        self._win.add_child(panel)
        self._panel = panel

        # Navigation row
        nav = gui.Horiz(6)
        btn_prev = gui.Button("◀ Prev")
        btn_prev.set_on_clicked(self._prev)
        btn_next = gui.Button("Next ▶")
        btn_next.set_on_clicked(self._next)
        self._lbl_idx = gui.Label(f"0 / {len(self._frames) - 1}")
        nav.add_child(btn_prev)
        nav.add_stretch()
        nav.add_child(self._lbl_idx)
        nav.add_stretch()
        nav.add_child(btn_next)
        panel.add_child(nav)
        panel.add_fixed(8)

        # Display mode
        panel.add_child(gui.Label("Display mode"))
        self._combo = gui.Combobox()
        for m in _BASE_MODES:
            self._combo.add_item(m)
        self._combo.selected_index = 0
        self._combo.set_on_selection_changed(lambda _t, _i: self._refresh())
        panel.add_child(self._combo)
        panel.add_fixed(12)

        # Model checkboxes
        panel.add_child(gui.Label("Models  (check to load)"))
        self._checkboxes: dict[str, gui.Checkbox] = {}
        all_ckpts = sorted(CKPT_BASE.glob("*/best.pth"))
        # pre-enabled models go first
        ordered = sorted(
            all_ckpts,
            key=lambda p: (p.parent.name not in self._predictors, p.parent.name),
        )
        for ckpt in ordered:
            name = ckpt.parent.name
            cb = gui.Checkbox(name)
            cb.checked = name in self._predictors
            if name in self._predictors:
                self._enabled[name] = True
            cb.set_on_checked(lambda checked, n=name, p=ckpt: self._toggle(n, p, checked))
            panel.add_child(cb)
            self._checkboxes[name] = cb
        panel.add_fixed(12)

        # Legend
        panel.add_child(gui.Label("Legend"))
        for label, hex_col in _LEGEND:
            r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
            lrow = gui.Horiz(4)
            dot = gui.Label("■")
            dot.text_color = gui.Color(r / 255, g / 255, b / 255)
            lrow.add_child(dot)
            lrow.add_child(gui.Label(f" {label}"))
            panel.add_child(lrow)

        # Add modes for pre-loaded predictors
        for name in list(self._predictors):
            self._enabled[name] = True
            self._combo.add_item(f"Model: {name}")
            self._combo.add_item(f"Model: {name} (vs GT)")

    # ── Layout ───────────────────────────────────────────────────────────────

    def _on_layout(self, _ctx) -> None:
        r = self._win.content_rect
        self._scene.frame = gui.Rect(r.x, r.y, r.width - _PANEL_W, r.height)
        self._panel.frame = gui.Rect(r.x + r.width - _PANEL_W, r.y, _PANEL_W, r.height)

    # ── Key handler ───────────────────────────────────────────────────────────

    def _on_key(self, ev: gui.KeyEvent) -> int:
        if ev.type == gui.KeyEvent.DOWN:
            if ev.key in (gui.KeyName.RIGHT, gui.KeyName.L):
                self._next()
                return gui.Widget.EventCallbackResult.HANDLED
            if ev.key in (gui.KeyName.LEFT, gui.KeyName.H):
                self._prev()
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    # ── Navigation ───────────────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._refresh()

    def _next(self) -> None:
        if self._idx < len(self._frames) - 1:
            self._idx += 1
            self._refresh()

    # ── Model toggle ──────────────────────────────────────────────────────────

    def _toggle(self, name: str, ckpt: Path, checked: bool) -> None:
        current_mode = self._combo.selected_text

        if checked:
            if name not in self._predictors:
                print(f"  [load] {name} …", flush=True)
                self._predictors[name] = ModelPredictor(ckpt)
            self._enabled[name] = True
            self._combo.add_item(f"Model: {name}")
            self._combo.add_item(f"Model: {name} (vs GT)")
            # auto-switch to this model
            for i in range(self._combo.number_of_items):
                if self._combo.get_item(i) == f"Model: {name}":
                    self._combo.selected_index = i
                    break
        else:
            self._enabled[name] = False
            # Remove model's combo entries; if one was selected, fall back to GT
            for item in (f"Model: {name}", f"Model: {name} (vs GT)"):
                if current_mode == item:
                    current_mode = _BASE_MODES[0]
                self._combo.remove_item(item)
            # Restore selection
            for i in range(self._combo.number_of_items):
                if self._combo.get_item(i) == current_mode:
                    self._combo.selected_index = i
                    break

        self._refresh()

    # ── Data helpers ─────────────────────────────────────────────────────────

    def _frame_data(self, idx: int) -> dict:
        if idx not in self._cache:
            pts, sem, trav = self._frames[idx].load()
            self._cache[idx] = {"pts": pts, "sem": sem, "trav": trav, "preds": {}}
        return self._cache[idx]

    def _prediction(self, idx: int, name: str) -> np.ndarray:
        data = self._frame_data(idx)
        if name not in data["preds"]:
            pred = self._predictors[name]
            print(f"  [infer] {name}  frame {idx} …", flush=True)
            data["preds"][name] = pred.predict(data["pts"])
        return data["preds"][name]

    # ── Render ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        data  = self._frame_data(self._idx)
        pts   = data["pts"]
        sem   = data["sem"]
        trav  = data["trav"]
        mode  = self._combo.selected_text

        # Resolve colours
        if mode == "GT traversability":
            colors = _color_gt(trav) if trav is not None else _color_intensity(pts)

        elif mode == "Semantic trav.":
            colors = _color_semantic(sem) if sem is not None else _color_intensity(pts)

        elif mode == "Composite (GT + Sem)":
            colors = _color_composite(trav, sem)

        elif mode == "Intensity":
            colors = _color_intensity(pts)

        elif mode.startswith("Model: ") and not mode.endswith("(vs GT)"):
            name = mode[len("Model: "):]
            if name in self._predictors and self._enabled.get(name):
                colors = _color_model(self._prediction(self._idx, name))
            else:
                colors = _color_intensity(pts)

        elif mode.endswith("(vs GT)"):
            m = re.match(r"Model: (.+) \(vs GT\)$", mode)
            name = m.group(1) if m else ""
            if name in self._predictors and self._enabled.get(name) and trav is not None:
                colors = _color_vs_gt(self._prediction(self._idx, name), trav)
            else:
                colors = _color_intensity(pts)

        else:
            colors = _color_intensity(pts)

        # Build point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

        sc = self._scene.scene
        sc.remove_geometry(self._GEOM)
        sc.add_geometry(self._GEOM, pcd, self._mat)

        self._lbl_idx.text = f"{self._idx} / {len(self._frames) - 1}"
        self._win.title = f"GOOSE Trav — {self._frames[self._idx].frame_id}"

        if self._idx == 0:
            bounds = sc.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._refresh()
        self._app.run()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ckpt(spec: str) -> Path:
    p = Path(spec)
    if p.exists():
        return p
    c = CKPT_BASE / spec / "best.pth"
    if c.exists():
        return c
    available = [x.parent.name for x in sorted(CKPT_BASE.glob("*/best.pth"))]
    raise FileNotFoundError(
        f"Checkpoint not found: '{spec}'\n  tried: {c}\n  available: {available}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",        default=GOOSE_ROOT,
                        help="GOOSE_3D root (contains train/ and val/).")
    parser.add_argument("--split",       default="val", choices=["train", "val"])
    parser.add_argument("--checkpoints", nargs="*", metavar="CKPT",
                        help="Experiment names or paths to pre-load.")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--start",       type=int, default=0)
    args = parser.parse_args()

    root   = Path(args.root)
    frames = build_frame_index(root, args.split)
    if not frames:
        print(f"No frames found under {root}/{args.split}")
        sys.exit(1)

    n_trav = sum(1 for f in frames if f.trav_path is not None)
    n_sem  = sum(1 for f in frames if f.sem_path  is not None)
    print(f"Frames: {len(frames)}  (trav_label: {n_trav}  sem_label: {n_sem})")

    predictors: list[ModelPredictor] = []
    for spec in (args.checkpoints or []):
        ckpt = _resolve_ckpt(spec)
        print(f"  Loading {ckpt.parent.name} …")
        predictors.append(ModelPredictor(ckpt, args.device))

    viewer = TravViewer(frames, predictors)
    viewer._idx = min(args.start, len(frames) - 1)
    viewer.run()


if __name__ == "__main__":
    main()
