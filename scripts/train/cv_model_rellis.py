"""Leave-one-sequence-out CV on RELLIS-3D for model architecture selection.

Compares SparseTravNet and MinkUNet at different channel ratios (cr),
using BCE as a fixed reference loss.  Aggregate mean ± std and Kendall's W
over folds are printed at the end.

Grid search is used (not Optuna) — the search space is small and discrete,
and fold-by-fold rankings are the primary output, not a single best value.

Usage:
    python -m scripts.train.cv_model_rellis --config resources/train_rellis_cv.yaml
    python -m scripts.train.cv_model_rellis --config resources/train_rellis_cv.yaml --folds 0 1
    python -m scripts.train.cv_model_rellis --config resources/train_rellis_cv.yaml --models travnet_cr05 minkunet_cr10
    python -m scripts.train.cv_model_rellis --config resources/train_rellis_cv.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets import RellisTorchDataset, filter_min_pos, sparse_collate
from src.losses.binary_class import TRAV_LOSSES
from src.metrics import BinaryMetrics, RankingMetrics, CalibrationMetrics
from src.models.sparse_trav_net import SparseTravNet
from src.models.torchsparse_minkunet import MinkUNet
from src.trainer import Trainer


# ---------------------------------------------------------------------------
# MinkUNet binary wrapper
# MinkUNet.forward returns {"encoded": ..., "output": (N, out_channels)}.
# The Trainer expects model(SparseTensor) -> (N,) logits tensor.
# ---------------------------------------------------------------------------

class _MinkUNetBinary(nn.Module):
    def __init__(self, in_channels: int, cr: float) -> None:
        super().__init__()
        self.net = MinkUNet(in_channels=in_channels, out_channels=1, cr=cr)

    def forward(self, x):
        return self.net(x)["output"].squeeze(-1)  # (N,)


# ---------------------------------------------------------------------------
# Experiment grid  (arch × cr)
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict] = [
    {"name": "travnet_cr025",  "arch": "travnet",  "cr": 0.25},
    {"name": "travnet_cr05",   "arch": "travnet",  "cr": 0.50},
    {"name": "travnet_cr10",   "arch": "travnet",  "cr": 1.00},
    {"name": "minkunet_cr025", "arch": "minkunet", "cr": 0.25},
    {"name": "minkunet_cr05",  "arch": "minkunet", "cr": 0.50},
    {"name": "minkunet_cr10",  "arch": "minkunet", "cr": 1.00},
]

BCE_CFG = {"name": "bce"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(exp: dict, in_channels: int, device: torch.device) -> nn.Module:
    arch, cr = exp["arch"], exp["cr"]
    if arch == "travnet":
        return SparseTravNet(in_channels=in_channels, cr=cr).to(device)
    elif arch == "minkunet":
        return _MinkUNetBinary(in_channels=in_channels, cr=cr).to(device)
    raise ValueError(f"Unknown arch: {arch}")


class _CachedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        print(f"    caching {len(dataset)} samples into RAM…", flush=True)
        self._data = [dataset[i] for i in range(len(dataset))]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int):
        return self._data[i]


def _filter_cached_min_pos(cached: "_CachedDataset", min_pos: int) -> Dataset:
    """Subset of an already-cached dataset — no data duplication in RAM."""
    if min_pos <= 0:
        return cached
    valid = [i for i in range(len(cached))
             if int((cached[i]["labels"] == 1).sum()) >= min_pos]
    n_skip = len(cached) - len(valid)
    if n_skip:
        print(f"[filter_min_pos] skipped {n_skip} scans (< {min_pos} positive labels)")
    return Subset(cached, valid)


def preload_all_sequences(cfg: dict) -> tuple[list[str], list[tuple[Dataset, Dataset]]]:
    """Load and cache all sequences once using Apairo's sequence API.

    Returns (seq_ids, per_seq) where per_seq[i] = (train_view, val_cached):
    - val_cached:  full sequence cached in RAM (one copy per sequence)
    - train_view:  lightweight Subset of val_cached with filter_min_pos applied
                   — no data duplication, train and val share the same backing cache
    """
    dc = cfg["data"]
    root = str(Path(dc["root"]).expanduser())
    min_pos = dc.get("min_pos", 1)

    full_ds = RellisTorchDataset(
        root_dir=root,
        voxel_size=dc["voxel_size"],
        voxel_size=dc["voxel_size"],
    )
    seq_ids = full_ds.sequence_ids  # discovered via Apairo, no manual fs scan

    per_seq: list[tuple[Dataset, Dataset]] = []
    for seq_id in seq_ids:
        print(f"  Loading sequence {seq_id}…", flush=True)
        cached     = _CachedDataset(full_ds.sequence(seq_id))
        train_view = _filter_cached_min_pos(cached, min_pos=min_pos)
        per_seq.append((train_view, cached))

    return seq_ids, per_seq


def build_fold_datasets(
    fold_idx: int,
    per_seq: list[tuple[Dataset, Dataset]],
) -> tuple[Dataset, Dataset]:
    train_ds = ConcatDataset([per_seq[i][0] for i in range(len(per_seq)) if i != fold_idx])
    val_ds   = per_seq[fold_idx][1]
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Single (fold × model) run — returns val metrics dict
# ---------------------------------------------------------------------------

def run_one(
    exp: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: dict,
    log_dir: str,
    save_dir: str,
) -> dict[str, float]:
    tc = cfg["training"]
    in_channels = cfg["model"].get("in_channels", 4)

    # Resume: if metrics.json exists the run completed — return cached results
    metrics_path = os.path.join(save_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            cached = json.load(f)
        print(f"    [SKIP] already done — loaded from {metrics_path}")
        return cached

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {}

    num_workers = tc.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        collate_fn=sparse_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        collate_fn=sparse_collate,
    )

    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(exp, in_channels, device)

    n_params = count_params(model)
    print(f"    params: {n_params:,}")

    criterion = TRAV_LOSSES[BCE_CFG["name"]](BCE_CFG).to(device)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=tc["epochs"],
        lr=tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-4),
        log_dir=log_dir,
        save_dir=save_dir,
        full_metrics=False,
    )
    trainer.fit()

    best_ckpt = os.path.join(save_dir, "best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    bm  = BinaryMetrics()
    rm  = RankingMetrics()
    cal = CalibrationMetrics()

    with torch.no_grad():
        for batch in val_loader:
            st      = batch["sparse_input"].to(device)
            traj_gt = batch["labels"].to(device)
            sem_gt  = batch.get("alt_labels")
            if st.feats.shape[0] == 0:
                continue
            try:
                logits = model(st)
            except Exception:
                continue
            eval_labels = sem_gt.to(device) if sem_gt is not None else traj_gt
            bm.update(logits, eval_labels)
            rm.update(logits, eval_labels)
            cal.update(logits, eval_labels)

    metrics = {**bm.compute(), **rm.compute(), **cal.compute(), "n_params": n_params}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# Ranking stability: Kendall's W
# ---------------------------------------------------------------------------

def kendalls_w(rank_matrix: np.ndarray) -> float:
    k, n = rank_matrix.shape
    if n < 2:
        return float("nan")
    R = rank_matrix.sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    return float(np.clip(W, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="resources/train_rellis_cv_model.yaml")
    parser.add_argument("--models",  nargs="+", help="Subset of experiment names")
    parser.add_argument("--folds",   nargs="+", type=int, help="Subset of fold indices (0-based)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Discover sequences via Apairo (lightweight — only path scanning, no data loaded)
    if not args.dry_run:
        print("\nPre-loading all sequences into RAM…")
        seq_ids, per_seq = preload_all_sequences(cfg)
    else:
        # Dry-run: discover sequences without loading data
        from src.datasets import RellisTorchDataset as _RDS
        _probe = _RDS(
            root_dir=str(Path(cfg["data"]["root"]).expanduser()),
            voxel_size=cfg["data"]["voxel_size"],
        )
        seq_ids = _probe.sequence_ids
        per_seq = []

    print(f"Found {len(seq_ids)} RELLIS sequences: {seq_ids}")

    if len(seq_ids) < 2:
        print("Need at least 2 sequences for cross-validation.")
        sys.exit(1)

    experiments = [e for e in EXPERIMENTS
                   if args.models is None or e["name"] in args.models]
    fold_indices = list(range(len(seq_ids)))
    if args.folds is not None:
        fold_indices = [i for i in args.folds if 0 <= i < len(seq_ids)]

    print(f"Models      : {[e['name'] for e in experiments]}")
    print(f"Loss (fixed): bce")
    print(f"Folds       : {fold_indices} (val sequences: {[seq_ids[i] for i in fold_indices]})")
    print(f"Epochs/run  : {cfg['training']['epochs']}")
    print(f"Total runs  : {len(experiments) * len(fold_indices)}")

    log_base  = Path(cfg["logging"]["log_dir"])
    save_base = Path(cfg["logging"]["save_dir"])
    out_dir   = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, dict]] = {i: {} for i in fold_indices}

    for fold_idx in fold_indices:
        val_seq   = seq_ids[fold_idx]
        train_ids = [s for s in seq_ids if s != val_seq]
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx}  —  val: {val_seq}  |  train: {train_ids}")
        print(f"{'='*70}")

        if not args.dry_run:
            train_ds, val_ds = build_fold_datasets(fold_idx, per_seq)

        for exp in experiments:
            print(f"\n  [{exp['name']}]  arch={exp['arch']}  cr={exp['cr']}")

            if args.dry_run:
                results[fold_idx][exp["name"]] = {"f1": 0.0, "iou": 0.0, "pr_auc": 0.0, "n_params": 0}
                continue

            log_dir  = str(log_base  / f"fold{fold_idx}_{val_seq}" / exp["name"])
            save_dir = str(save_base / f"fold{fold_idx}_{val_seq}" / exp["name"])
            os.makedirs(log_dir,  exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            try:
                metrics = run_one(exp, train_ds, val_ds, cfg, log_dir, save_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                metrics = {}

            results[fold_idx][exp["name"]] = metrics

            if metrics:
                print(
                    f"  -> f1={metrics.get('f1', float('nan')):.4f}"
                    f"  iou={metrics.get('iou', float('nan')):.4f}"
                    f"  pr_auc={metrics.get('pr_auc', float('nan')):.4f}"
                    f"  coll_fpr={metrics.get('collision_fpr', float('nan')):.4f}"
                    f"  ece={metrics.get('ece', float('nan')):.4f}"
                    f"  params={metrics.get('n_params', 0):,}"
                )

    # ── Aggregate ─────────────────────────────────────────────────────────────

    KEY = "f1"

    agg: dict[str, dict] = {}
    for exp in experiments:
        vals = [results[i][exp["name"]].get(KEY, float("nan"))
                for i in fold_indices
                if results[i].get(exp["name"])]
        vals = [v for v in vals if not np.isnan(v)]
        # n_params is constant across folds — take first available
        n_params = next(
            (results[i][exp["name"]].get("n_params", 0)
             for i in fold_indices if results[i].get(exp["name"])),
            0,
        )
        agg[exp["name"]] = {
            "mean":     float(np.mean(vals)) if vals else float("nan"),
            "std":      float(np.std(vals))  if vals else float("nan"),
            "n":        len(vals),
            "n_params": n_params,
        }

    rank_matrix = np.zeros((len(fold_indices), len(experiments)))
    for fi, fold_idx in enumerate(fold_indices):
        fold_vals = [(exp["name"], results[fold_idx].get(exp["name"], {}).get(KEY, float("nan")))
                     for exp in experiments]
        sorted_names = [n for n, _ in sorted(fold_vals, key=lambda x: -x[1])]
        for ei, exp in enumerate(experiments):
            rank = sorted_names.index(exp["name"]) + 1 if exp["name"] in sorted_names else len(experiments)
            rank_matrix[fi, ei] = rank

    w = kendalls_w(rank_matrix)

    # ── Print summary table ────────────────────────────────────────────────────

    sorted_exps = sorted(experiments, key=lambda e: agg[e["name"]]["mean"], reverse=True)
    W = max(len(e["name"]) for e in experiments)

    print(f"\n{'='*90}")
    print(f"  MODEL CV SUMMARY  ({len(fold_indices)} folds, loss=bce, metric={KEY})")
    print(f"  Kendall's W = {w:.3f}  {'(stable ✓)' if w > 0.7 else '(unstable — more data needed)'}")
    print(f"{'='*90}")
    fold_headers = "  ".join(f"fold{i:1d}" for i in fold_indices)
    print(f"{'Model':<{W}}  {'Params':>9}  {'Mean':>7}  {'Std':>6}  {fold_headers}")
    print("-" * 90)

    for exp in sorted_exps:
        name = exp["name"]
        m = agg[name]
        fold_vals_str = "  ".join(
            f"{results[i].get(name, {}).get(KEY, float('nan')):.3f}"
            for i in fold_indices
        )
        print(f"{name:<{W}}  {m['n_params']:>9,}  {m['mean']:>7.4f}  {m['std']:>6.4f}  {fold_vals_str}")

    # ── Save results ───────────────────────────────────────────────────────────

    out_json = out_dir / "rellis_cv_model_results.json"
    out_csv  = out_dir / "rellis_cv_model_results.csv"

    full = {
        "kendalls_w": w,
        "loss":       "bce",
        "folds":      {seq_ids[i]: {e["name"]: results[i].get(e["name"], {})
                                    for e in experiments}
                       for i in fold_indices},
        "aggregate":  agg,
    }
    with open(out_json, "w") as f:
        json.dump(full, f, indent=2)

    fields = ["model", "arch", "cr", "n_params", "mean_f1", "std_f1"] + [f"fold{i}_f1" for i in fold_indices]
    rows = []
    for exp in sorted_exps:
        row = {
            "model":    exp["name"],
            "arch":     exp["arch"],
            "cr":       exp["cr"],
            "n_params": agg[exp["name"]]["n_params"],
            "mean_f1":  agg[exp["name"]]["mean"],
            "std_f1":   agg[exp["name"]]["std"],
        }
        for i in fold_indices:
            row[f"fold{i}_f1"] = results[i].get(exp["name"], {}).get(KEY, float("nan"))
        rows.append(row)

    with open(out_csv, "w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w_csv.writeheader()
        w_csv.writerows(rows)

    print(f"\nSaved → {out_json}  /  {out_csv}")
    print(f"\nKendall's W = {w:.3f}  |  interpretation:")
    print(f"  W > 0.7  → rankings stable across folds")
    print(f"  W > 0.5  → moderate agreement")
    print(f"  W < 0.5  → high variance, rankings unreliable")


if __name__ == "__main__":
    main()
