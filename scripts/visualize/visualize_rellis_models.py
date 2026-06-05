"""Compare trained models on a RELLIS-3D sequence with Rerun.

One viewport for ground truth (trav_gt ⊕ trav_label), one per model checkpoint.

Usage:
    python -m scripts.visualize.visualize_rellis_models
    python -m scripts.visualize.visualize_rellis_models --config resources/visualize_rellis_models.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

import apairo_rr
from apairo_rr import Pipeline, load_label_config
from apairo import Rellis3DDataset
from src.datasets import RellisCompositeDataset

GT_CFG = {
    "color_map": {
        0: [128, 128, 128],
        1: [39,  174,  96],
        2: [41,  128, 185],
        3: [244, 208,  63],
    },
    "semantic_map": {
        0: "non-traversable",
        1: "traj-GT only",
        2: "sem-label only",
        3: "both agree",
    },
}

MODEL_CFG = {
    "color_map":    {0: [200, 60, 60], 1: [50, 200, 80]},
    "semantic_map": {0: "not traversable", 1: "traversable"},
}


def _build_model(arch: str, cr: float, checkpoint: Path, device: str) -> torch.nn.Module:
    from src.models.sparse_trav_net import SparseTravNet
    from src.models.torchsparse_minkunet import MinkUNet

    if arch == "travnet":
        model = SparseTravNet(in_channels=4, cr=cr).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    elif arch == "minkunet":
        model = MinkUNet(in_channels=4, out_channels=1, cr=cr).to(device)
        # Checkpoints were saved from _MinkUNetBinary (self.net = MinkUNet),
        # so state dict keys have a "net." prefix — strip it before loading.
        sd = torch.load(checkpoint, map_location=device, weights_only=True)
        sd = {k.removeprefix("net."): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.eval()
    return model


def make_inference_step(spec: dict, device: str, voxel_size: float, max_rad: float):
    from torchsparse import SparseTensor
    from torchsparse.utils.quantize import sparse_quantize

    arch  = spec.get("arch", "travnet")
    cr    = spec["cr"]
    ckpt  = Path(spec["checkpoint"])
    model = _build_model(arch, cr, ckpt, device)

    def _infer(pts: np.ndarray, labels: np.ndarray | None):
        xyz       = pts[:, :3].astype(np.float32)
        intensity = pts[:, 3].astype(np.float32)
        mask      = np.linalg.norm(xyz, axis=1) < max_rad
        xyz_f, inten_f = xyz[mask], intensity[mask]
        if len(xyz_f) == 0:
            return pts, np.zeros(len(pts), dtype=np.int64)

        coords_q = np.floor(xyz_f / voxel_size).astype(np.int32)
        coords_q, sel, inv = sparse_quantize(coords_q, return_index=True, return_inverse=True)
        feats = np.column_stack([xyz_f[sel], inten_f[sel]])
        bc    = np.hstack([np.zeros((len(coords_q), 1), dtype=np.int32), coords_q])

        st = SparseTensor(
            coords=torch.from_numpy(bc).int(),
            feats=torch.from_numpy(feats).float(),
        ).to(device)
        with torch.no_grad():
            logits = model(st)
            if isinstance(logits, dict):
                logits = logits["output"].squeeze(-1)
            pred_vox = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int64)

        pred_pts       = np.zeros(len(pts), dtype=np.int64)
        pred_pts[mask] = pred_vox[inv]
        return pts, pred_pts

    return _infer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/visualize_rellis_models.yaml")
    parser.add_argument("--web", default=False, action="store_true",
                        help="Serve web viewer (streams, subject to memory limit)")
    parser.add_argument("--rrd", default=None, metavar="PATH",
                        help="Save recording to .rrd file instead of streaming. "
                             "Open afterwards with: rerun <PATH>  or  rerun --web-viewer <PATH>")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    web = args.web
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dc  = cfg["data"]
    ic  = cfg.get("inference", {})
    root       = Path(dc["root"]).expanduser()
    seq_id     = str(dc["sequence_id"])
    every      = dc.get("every", 1)
    device     = ic.get("device", "cpu")
    voxel_size = ic.get("voxel_size", 0.1)
    max_rad    = ic.get("max_rad", 50.0)

    dataset = RellisCompositeDataset(root)
    print(f"Dataset: {len(dataset)} scans total")

    frames = list(dataset.sequence(seq_id)._indices)[::every]
    print(f"Sequence {seq_id}: {len(frames)} frames (every={every})")

    # Pipeline GT — traversabilité composite (trav_gt ⊕ trav_label)
    pipelines  = [Pipeline("GT-Trav")]
    label_cfgs = [GT_CFG]

    # Pipeline GT — labels sémantiques RELLIS bruts
    sem_ds = Rellis3DDataset(root_dir=str(root), keys=["lidar", "labels"])
    sem_labels = [np.asarray(sem_ds[i].data["labels"]) for i in frames]

    class _SemanticStep:
        def __init__(self, labels_list: list):
            self._it = iter(labels_list)
        def __call__(self, pts: np.ndarray, _labels):
            return pts, next(self._it)

    pipelines.append(Pipeline("GT-Semantic", [_SemanticStep(sem_labels)]))
    label_cfgs.append(load_label_config("rellis"))

    global_cr = ic.get("cr")  # backward-compat: old configs set cr once in inference block

    for spec in cfg.get("models", []):
        ckpt = Path(spec["checkpoint"])
        if not ckpt.is_absolute():
            ckpt = _ROOT / ckpt
        if not ckpt.exists():
            print(f"[warn] checkpoint not found: {ckpt}")
            continue
        spec = dict(spec)  # don't mutate the parsed YAML
        spec["checkpoint"] = str(ckpt)
        spec.setdefault("arch", "travnet")
        # per-model cr takes priority; fall back to global inference.cr
        if "cr" not in spec:
            if global_cr is None:
                print(f"[warn] no cr for {spec['name']} — skipping")
                continue
            spec["cr"] = global_cr
        print(f"  Loading: {spec['name']}  arch={spec['arch']}  cr={spec['cr']}  ({ckpt})")
        pipelines.append(Pipeline(spec["name"], [
            make_inference_step(spec, device, voxel_size, max_rad)
        ]))
        label_cfgs.append(MODEL_CFG)

    if args.rrd is not None:
        import rerun as rr
        rrd_path = Path(args.rrd)
        rrd_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving recording to {rrd_path} …")
        print(f"Open with:  rerun {rrd_path}")
        print(f"  or web:   rerun --web-viewer {rrd_path}")

        # apairo_rr.view appelle rr.init() en interne avant tout log.
        # On patche rr.init pour injecter rr.save() juste après.
        _orig_init = rr.init
        def _init_and_save(*a, **kw):
            kw["spawn"] = False   # ne pas ouvrir le viewer
            _orig_init(*a, **kw)
            rr.save(str(rrd_path))
        rr.init = _init_and_save

    apairo_rr.view(
        dataset,
        label_cfgs=label_cfgs,
        label_key="trav_composite",
        point_key="lidar",
        pipelines=pipelines,
        frames=frames,
        application_id="rellis_model_comparison",
        web=web,
        spawn=args.rrd is None and not web,
    )

    if args.rrd is not None:
        rr.init = _orig_init   # restore


if __name__ == "__main__":
    main()
