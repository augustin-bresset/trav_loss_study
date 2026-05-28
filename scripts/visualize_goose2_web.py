"""Web-based GOOSE-3D traversability viewer (Dash + Plotly, no apairo).

Accessible depuis un navigateur sur le même réseau :
    http://<machine-ip>:8050

Lit directement les .bin / .label / .npy depuis le filesystem GOOSE.
Charge les modèles en lazy loading quand leur mode est sélectionné.

Display modes (dropdown) :
  GT traversability      — vert / gris
  Semantic trav.         — violet / gris  (classes GOOSE traversables)
  Composite (GT + Sem)   — carte 4 couleurs
  Intensity              — niveaux de gris
  Model: <name>          — orange / gris  (ajouté quand ☑ model)
  Model: <name> (vs GT)  — TP/FP/FN/TN    (ajouté quand ☑ model)

Navigation : boutons Prev / Next ou champ numérique.

Usage :
    python scripts/visualize_goose2_web.py
    python scripts/visualize_goose2_web.py --port 8050 --host 0.0.0.0
    python scripts/visualize_goose2_web.py --start 42 --max-pts 30000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from src.models.sparse_trav_net import SparseTravNet

# ── constants ─────────────────────────────────────────────────────────────────
GOOSE_ROOT   = "/mnt/vault-fellowship/goose/GOOSE_3D"
CKPT_BASE    = _ROOT / "data" / "checkpoints" / "goose"
VOXEL_SIZE   = 0.1
MAX_RAD      = 50.0
TRAV_SEM_IDS = frozenset({23, 24, 31, 50, 51})
MAX_PTS      = 40_000   # points affichés max (perf navigateur)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_FRAME_RE = re.compile(r"^(.+?)(?:_pcl|_goose)$")

# Couleurs hex pour Plotly
_COL = {
    "gt_pos":  "#27AE60",
    "sem_pos": "#8E44AD",
    "both":    "#F4D03F",
    "model":   "#E67E22",
    "tp":      "#27AE60",
    "fp":      "#295BA8",
    "fn":      "#C41E3A",
    "neg":     "#6B6B6B",
}
_COMPOSITE_COLS = {0: "#6B6B6B", 1: "#27AE60", 2: "#8E44AD", 3: "#F4D03F"}

_BASE_MODES = [
    "GT traversability",
    "Semantic trav.",
    "Composite (GT + Sem)",
    "Intensity",
]

# ─────────────────────────────────────────────────────────────────────────────
# Frame index
# ─────────────────────────────────────────────────────────────────────────────

class Frame:
    def __init__(self, lidar: Path, sem: Path | None, trav: Path | None, fid: str):
        self.lidar_path = lidar
        self.sem_path   = sem
        self.trav_path  = trav
        self.frame_id   = fid

    def load(self):
        pts  = np.fromfile(self.lidar_path, dtype=np.float32).reshape(-1, 4)
        sem  = np.fromfile(self.sem_path,   dtype=np.int32)               if self.sem_path  else None
        trav = np.load(self.trav_path).astype(np.int32)                   if self.trav_path else None
        return pts, sem, trav


def build_frame_index(root: Path, split: str) -> list[Frame]:
    split_root = root / split

    def _idx(paths):
        d = {}
        for p in sorted(paths):
            m = _FRAME_RE.match(p.stem)
            if m:
                d[m.group(1)] = p
        return d

    lidar = _idx(split_root.rglob("*_pcl.bin"))
    sem   = _idx(split_root.rglob("*_goose.label"))
    trav  = _idx(split_root.rglob("*_goose.npy"))

    return [Frame(lp, sem.get(f), trav.get(f), f) for f, lp in sorted(lidar.items())]


# ─────────────────────────────────────────────────────────────────────────────
# Caches (module-level, partagés entre callbacks Dash)
# ─────────────────────────────────────────────────────────────────────────────

_frame_cache: dict[int, dict]          = {}   # idx → {pts, sem, trav, preds}
_model_cache: dict[str, SparseTravNet] = {}   # name → model

def _load_frame(frames: list[Frame], idx: int) -> dict:
    if idx not in _frame_cache:
        pts, sem, trav = frames[idx].load()
        _frame_cache[idx] = {"pts": pts, "sem": sem, "trav": trav, "preds": {}}
    return _frame_cache[idx]


def _load_model(name: str, device: str = DEFAULT_DEVICE) -> SparseTravNet:
    if name not in _model_cache:
        ckpt = CKPT_BASE / name / "best.pth"
        print(f"  [load] {name} …", flush=True)
        net = SparseTravNet(in_channels=4, cr=1.0).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        net.eval()
        _model_cache[name] = net
    return _model_cache[name]


def _predict(data: dict, name: str, device: str = DEFAULT_DEVICE) -> np.ndarray:
    if name not in data["preds"]:
        from torchsparse import SparseTensor
        from torchsparse.utils.quantize import sparse_quantize

        pts   = data["pts"]
        xyz   = pts[:, :3].astype(np.float32)
        inten = pts[:, 3].astype(np.float32)
        mask  = np.linalg.norm(xyz, axis=1) < MAX_RAD
        xyz_f, inten_f = xyz[mask], inten[mask]

        if len(xyz_f) == 0:
            data["preds"][name] = np.zeros(len(pts), dtype=np.int32)
            return data["preds"][name]

        net = _load_model(name, device)
        coords_q = np.floor(xyz_f / VOXEL_SIZE).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)
        feats_q = np.column_stack([xyz_f[sel], inten_f[sel]])
        batch_c = np.hstack([np.zeros((len(coords_q), 1), dtype=np.int32), coords_q])

        st = SparseTensor(
            coords=torch.from_numpy(batch_c).int(),
            feats=torch.from_numpy(feats_q).float(),
        ).to(device)

        with torch.no_grad():
            pred_vox = (torch.sigmoid(net(st)) > 0.5).cpu().numpy().astype(np.int32)

        pred = np.zeros(len(pts), dtype=np.int32)
        pred[mask] = pred_vox[inv]
        data["preds"][name] = pred
        print(f"  [infer] {name}  → {pred.sum()} pos / {len(pred)}", flush=True)

    return data["preds"][name]


# ─────────────────────────────────────────────────────────────────────────────
# Colourisation
# ─────────────────────────────────────────────────────────────────────────────

def _colorize(mode: str, pts, sem, trav, pred=None) -> list[str]:
    n = len(pts)

    if mode == "GT traversability":
        if trav is None:
            return [_COL["neg"]] * n
        return [_COL["gt_pos"] if t else _COL["neg"] for t in trav]

    if mode == "Semantic trav.":
        if sem is None:
            return [_COL["neg"]] * n
        return [_COL["sem_pos"] if s in TRAV_SEM_IDS else _COL["neg"] for s in sem]

    if mode == "Composite (GT + Sem)":
        bits = np.zeros(n, dtype=np.int32)
        if trav is not None:
            bits |= (trav == 1).astype(np.int32)
        if sem is not None:
            bits |= np.isin(sem, list(TRAV_SEM_IDS)).astype(np.int32) << 1
        return [_COMPOSITE_COLS.get(int(b), _COL["neg"]) for b in bits]

    if mode == "Intensity":
        v = pts[:, 3]
        v = (v - v.min()) / (v.max() - v.min() + 1e-6)
        def _gray(x): h = int(x * 255); return f"#{h:02x}{h:02x}{h:02x}"
        return [_gray(float(x)) for x in v]

    if mode.endswith("(vs GT)") and pred is not None and trav is not None:
        cols = []
        for p, t in zip(pred, trav):
            if t and p:     cols.append(_COL["tp"])
            elif not t and p: cols.append(_COL["fp"])
            elif t and not p: cols.append(_COL["fn"])
            else:             cols.append(_COL["neg"])
        return cols

    if pred is not None:  # "Model: <name>"
        return [_COL["model"] if p else _COL["neg"] for p in pred]

    return [_COL["neg"]] * n


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figure builder
# ─────────────────────────────────────────────────────────────────────────────

def _initial_figure() -> go.Figure:
    """Figure vide avec le layout fixe — initialisée une seule fois dans dcc.Graph."""
    fig = go.Figure(go.Scatter3d(
        x=[], y=[], z=[],
        mode="markers",
        marker=dict(size=1.5, color=[], opacity=0.85),
        hoverinfo="skip",
    ))
    fig.update_layout(
        uirevision="stable",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
            aspectmode="data",
            bgcolor="#111111",
        ),
        paper_bgcolor="#1a1a1a",
        font=dict(color="#dddddd"),
    )
    return fig


def _patch_figure(pts, colors_all: list[str], max_pts: int):
    """Retourne un Patch qui met à jour uniquement les données — préserve la caméra."""
    from dash import Patch
    n = len(pts)
    idx = np.random.choice(n, max_pts, replace=False) if n > max_pts else np.arange(n)

    patched = Patch()
    patched["data"][0]["x"] = pts[idx, 0].tolist()
    patched["data"][0]["y"] = pts[idx, 1].tolist()
    patched["data"][0]["z"] = pts[idx, 2].tolist()
    patched["data"][0]["marker"]["color"] = [colors_all[i] for i in idx]
    return patched


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def make_app(frames: list[Frame], device: str, max_pts: int) -> dash.Dash:
    n_frames  = len(frames)
    all_names = sorted(p.parent.name for p in CKPT_BASE.glob("*/best.pth"))

    app = dash.Dash(__name__, title="GOOSE Traversability")
    app.layout = html.Div([
        # ── état courant ────────────────────────────────────────────────────
        dcc.Store(id="frame-idx", data=0),
        dcc.Store(id="active-models-prev", data=[]),

        # ── layout global ───────────────────────────────────────────────────
        html.Div(style={"display": "flex", "height": "100vh",
                        "backgroundColor": "#1a1a1a", "color": "#ddd",
                        "fontFamily": "monospace"}, children=[

            # ── 3D graph ────────────────────────────────────────────────────
            html.Div(style={"flex": "1", "minWidth": 0, "position": "relative"}, children=[
                html.Div(id="frame-title", style={
                    "position": "absolute", "top": "8px", "left": "12px",
                    "zIndex": "10", "fontSize": "11px", "color": "#aaa",
                    "pointerEvents": "none",
                }),
                dcc.Graph(
                    id="graph-3d",
                    figure=_initial_figure(),
                    style={"height": "100%"},
                    config={"scrollZoom": True, "displaylogo": False},
                ),
            ]),

            # ── panneau droit ────────────────────────────────────────────────
            html.Div(style={
                "width": "280px", "flexShrink": "0", "overflowY": "auto",
                "padding": "16px", "backgroundColor": "#252525",
                "borderLeft": "1px solid #444",
            }, children=[

                # Navigation
                html.Div("Navigation", style={"fontWeight": "bold", "marginBottom": "8px"}),
                html.Div(style={"display": "flex", "alignItems": "center",
                                "gap": "8px", "marginBottom": "12px"}, children=[
                    html.Button("◀", id="btn-prev", n_clicks=0,
                                style=_btn_style()),
                    dcc.Input(id="frame-input", type="number", min=0,
                              max=n_frames - 1, step=1, value=0,
                              style={"width": "60px", "textAlign": "center",
                                     "backgroundColor": "#333", "color": "#ddd",
                                     "border": "1px solid #555", "borderRadius": "4px",
                                     "padding": "4px"}),
                    html.Span(f"/ {n_frames - 1}", style={"color": "#aaa"}),
                    html.Button("▶", id="btn-next", n_clicks=0,
                                style=_btn_style()),
                ]),

                html.Hr(style={"borderColor": "#444"}),

                # Display mode
                html.Div("Display mode", style={"fontWeight": "bold", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="mode-dropdown",
                    options=[{"label": m, "value": m} for m in _BASE_MODES],
                    value=_BASE_MODES[0],
                    clearable=False,
                    style={"backgroundColor": "#333", "color": "#111",
                           "marginBottom": "12px"},
                ),

                html.Hr(style={"borderColor": "#444"}),

                # Models
                html.Div("Models  (☑ = charger)", style={"fontWeight": "bold",
                                                          "marginBottom": "8px"}),
                dcc.Checklist(
                    id="model-checklist",
                    options=[{"label": f"  {n}", "value": n} for n in all_names],
                    value=[],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "block", "marginBottom": "4px",
                                "fontSize": "12px"},
                ),

                html.Hr(style={"borderColor": "#444"}),

                # Légende
                html.Div("Légende", style={"fontWeight": "bold", "marginBottom": "6px"}),
                *[
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "marginBottom": "3px", "fontSize": "12px"}, children=[
                        html.Div(style={"width": "12px", "height": "12px",
                                        "backgroundColor": col, "marginRight": "8px",
                                        "flexShrink": "0", "borderRadius": "2px"}),
                        html.Span(label),
                    ])
                    for label, col in [
                        ("GT traversable",      _COL["gt_pos"]),
                        ("Semantic traversable",_COL["sem_pos"]),
                        ("GT + Semantic",        _COL["both"]),
                        ("Model prediction",    _COL["model"]),
                        ("TP (GT & model)",      _COL["tp"]),
                        ("FP (model seulement)", _COL["fp"]),
                        ("FN (GT seulement)",    _COL["fn"]),
                        ("Non-traversable",      _COL["neg"]),
                    ]
                ],
            ]),
        ]),
    ], style={"margin": 0, "padding": 0})

    # ── Callback 1 : navigation ──────────────────────────────────────────────
    @app.callback(
        Output("frame-idx",   "data"),
        Output("frame-input", "value"),
        Input("btn-prev",    "n_clicks"),
        Input("btn-next",    "n_clicks"),
        Input("frame-input", "value"),
        State("frame-idx",   "data"),
        prevent_initial_call=True,
    )
    def navigate(n_prev, n_next, input_val, current):
        triggered = ctx.triggered_id
        if triggered == "btn-prev":
            new = max(0, current - 1)
        elif triggered == "btn-next":
            new = min(n_frames - 1, current + 1)
        else:
            new = max(0, min(n_frames - 1, int(input_val or 0)))
        return new, new

    # ── Callback 2 : dropdown options selon modèles cochés ──────────────────
    @app.callback(
        Output("mode-dropdown",    "options"),
        Output("mode-dropdown",    "value"),
        Output("active-models-prev", "data"),
        Input("model-checklist",   "value"),
        State("mode-dropdown",     "value"),
        State("active-models-prev", "data"),
    )
    def update_modes(active, current_mode, prev_active):
        prev_set = set(prev_active or [])
        curr_set = set(active or [])
        newly_added = curr_set - prev_set

        options = [{"label": m, "value": m} for m in _BASE_MODES]
        for name in sorted(curr_set):
            options.append({"label": f"Model: {name}",         "value": f"Model: {name}"})
            options.append({"label": f"Model: {name} (vs GT)", "value": f"Model: {name} (vs GT)"})

        valid_vals = {o["value"] for o in options}
        new_mode = current_mode if current_mode in valid_vals else _BASE_MODES[0]

        # Auto-sélectionner le dernier modèle ajouté
        if newly_added:
            new_mode = f"Model: {sorted(newly_added)[-1]}"

        return options, new_mode, list(curr_set)

    # ── Callback 3 : rendu du nuage (Patch = caméra préservée) ──────────────
    @app.callback(
        Output("graph-3d",    "figure"),
        Output("frame-title", "children"),
        Input("frame-idx",       "data"),
        Input("mode-dropdown",   "value"),
        Input("model-checklist", "value"),
    )
    def update_graph(frame_idx, mode, active_models):
        data = _load_frame(frames, frame_idx)
        pts, sem, trav = data["pts"], data["sem"], data["trav"]

        pred = None
        m = re.match(r"Model: (.+?)(?:\s+\(vs GT\))?$", mode or "")
        if m:
            name = m.group(1)
            if name in (active_models or []):
                pred = _predict(data, name, device)

        colors = _colorize(mode or _BASE_MODES[0], pts, sem, trav, pred)
        fid    = frames[frame_idx].frame_id
        return _patch_figure(pts, colors, max_pts), fid

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers CSS
# ─────────────────────────────────────────────────────────────────────────────

def _btn_style() -> dict:
    return {
        "backgroundColor": "#3a3a3a", "color": "#ddd",
        "border": "1px solid #555", "borderRadius": "4px",
        "padding": "4px 10px", "cursor": "pointer",
        "fontSize": "16px",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",     default=GOOSE_ROOT)
    parser.add_argument("--split",    default="val", choices=["train", "val"])
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     type=int, default=8050)
    parser.add_argument("--device",   default=DEFAULT_DEVICE)
    parser.add_argument("--start",    type=int, default=0)
    parser.add_argument("--max-pts",  type=int, default=MAX_PTS,
                        help="Points max affichés par frame (défaut 40 000).")
    args = parser.parse_args()

    root   = Path(args.root)
    frames = build_frame_index(root, args.split)
    if not frames:
        print(f"Aucune frame trouvée sous {root}/{args.split}")
        sys.exit(1)

    n_trav = sum(1 for f in frames if f.trav_path)
    n_sem  = sum(1 for f in frames if f.sem_path)
    print(f"Frames : {len(frames)}  (trav_label : {n_trav}  sem_label : {n_sem})")
    print(f"Serveur : http://{args.host}:{args.port}")
    print("Accessible depuis le réseau via l'IP de cette machine.\n")

    app = make_app(frames, args.device, args.max_pts)

    # Pré-positionner sur la frame de départ
    app.layout.children[0].data = args.start  # frame-store

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
