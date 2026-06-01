"""Leave-one-sequence-out cross-validation on RELLIS-3D.

For each of the 5 RELLIS sequences used as validation in turn, trains all
losses and records metrics.  Aggregate mean ± std and Kendall's W (ranking
concordance across folds) are printed at the end.

A high Kendall's W (> 0.7) means the loss rankings are stable across
sequences — i.e. your results are trustworthy.

Usage:
    python -m scripts.train.cv_rellis --config resources/train_rellis_cv.yaml
    python -m scripts.train.cv_rellis --config resources/train_rellis_cv.yaml --dry-run
    python -m scripts.train.cv_rellis --config resources/train_rellis_cv.yaml --losses bce focal nnpu_p30
    python -m scripts.train.cv_rellis --config resources/train_rellis_cv.yaml --folds 0 1 2
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
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))

from apairo import Rellis3DDataset, split_sequences
from src.datasets import RellisTorchDataset, sparse_collate
from src.losses.binary_class import TRAV_LOSSES
from src.metrics import BinaryMetrics, RankingMetrics, CalibrationMetrics
from src.models.sparse_trav_net import SparseTravNet
from src.trainer import Trainer


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict] = [
    # ── Baselines ─────────────────────────────────────────────────────────
    {"name": "bce",           "loss": {"name": "bce"}},
    {"name": "focal",         "loss": {"name": "focal",  "gamma": 2.0, "pos_weight": 3.6}},
    # ── Imbalance / focus ─────────────────────────────────────────────────
    {"name": "asl",           "loss": {"name": "asl",    "gamma_neg": 4.0, "asl_clip": 0.05}},
    {"name": "tversky",       "loss": {"name": "tversky","tversky_alpha": 0.3, "tversky_beta": 0.7}},
    {"name": "focal_tversky", "loss": {"name": "focal_tversky", "tversky_alpha": 0.3, "tversky_beta": 0.7}},
    {"name": "lovasz",        "loss": {"name": "lovasz"}},
    # ── Hybrid ────────────────────────────────────────────────────────────
    {"name": "bce_dice",      "loss": {"name": "bce_dice"}},
    {"name": "bce_lovasz",    "loss": {"name": "bce_lovasz"}},
    {"name": "focal_dice",    "loss": {"name": "focal_dice", "gamma": 2.0}},
    # ── Noise-robust ──────────────────────────────────────────────────────
    {"name": "sce",           "loss": {"name": "sce",    "sce_alpha": 0.1, "sce_beta": 1.0}},
    {"name": "gce",           "loss": {"name": "gce",    "gce_q": 0.7}},
    # ── PU / weak supervision ─────────────────────────────────────────────
    {"name": "nnpu_p20",      "loss": {"name": "bce_nnpu", "prior": 0.20}},
    {"name": "nnpu_p30",      "loss": {"name": "bce_nnpu", "prior": 0.30}},
    {"name": "nnpu_p40",      "loss": {"name": "bce_nnpu", "prior": 0.40}},
    {"name": "focal_nnpu_p30","loss": {"name": "focal_nnpu","prior": 0.30, "gamma": 2.0}},
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Single (fold × loss) run — returns val metrics dict
# ---------------------------------------------------------------------------

def run_one(
    exp: dict,
    train_ids: list[str],
    val_ids: list[str],
    cfg: dict,
    log_dir: str,
    save_dir: str,
) -> dict[str, float]:
    dc = cfg["data"]
    tc = cfg["training"]
    mc = cfg["model"]

    train_ds = RellisTorchDataset(
        root_dir=dc["root"],
        sequence_ids=train_ids,
        voxel_size=dc["voxel_size"],
        max_rad=dc["max_rad"],
        min_pos=dc.get("min_pos", 1),
    )
    val_ds = RellisTorchDataset(
        root_dir=dc["root"],
        sequence_ids=val_ids,
        voxel_size=dc["voxel_size"],
        max_rad=dc["max_rad"],
        min_pos=0,  # keep all val scans regardless of label density
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {}

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc.get("num_workers", 4), pin_memory=True,
        collate_fn=sparse_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc.get("num_workers", 4), pin_memory=True,
        collate_fn=sparse_collate,
    )

    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")
    model  = SparseTravNet(
        in_channels=mc.get("in_channels", 4),
        cr=mc.get("cr", 1.0),
    ).to(device)

    criterion = TRAV_LOSSES[exp["loss"]["name"]](exp["loss"]).to(device)

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
    )
    trainer.fit()

    # Final snapshot — load best checkpoint and evaluate
    best_ckpt = os.path.join(save_dir, "best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    bm  = BinaryMetrics()
    rm  = RankingMetrics()
    cal = CalibrationMetrics()

    with torch.no_grad():
        for batch in val_loader:
            st        = batch["sparse_input"].to(device)
            traj_gt   = batch["labels"].to(device)
            sem_gt    = batch.get("alt_labels")  # trav_label — semantic GT
            if st.feats.shape[0] == 0:
                continue
            try:
                logits = model(st)
            except Exception:
                continue
            # Evaluate against semantic GT when available (cleaner reference)
            eval_labels = sem_gt.to(device) if sem_gt is not None else traj_gt
            bm.update(logits, eval_labels)
            rm.update(logits, eval_labels)
            cal.update(logits, eval_labels)

    return {**bm.compute(), **rm.compute(), **cal.compute()}


# ---------------------------------------------------------------------------
# Ranking stability: Kendall's W
# ---------------------------------------------------------------------------

def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W (coefficient of concordance) for (n_folds × n_losses) rank matrix."""
    k, n = rank_matrix.shape
    if n < 2:
        return float("nan")
    R    = rank_matrix.sum(axis=0)
    S    = ((R - R.mean()) ** 2).sum()
    W    = 12 * S / (k ** 2 * (n ** 3 - n))
    return float(np.clip(W, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="resources/train_rellis_cv.yaml")
    parser.add_argument("--losses",  nargs="+", help="Subset of experiment names")
    parser.add_argument("--folds",   nargs="+", type=int, help="Subset of fold indices (0-based)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Discover RELLIS sequences
    probe   = Rellis3DDataset(cfg["data"]["root"], keys=["lidar"])
    seq_ids = probe.sequence_ids
    print(f"Found {len(seq_ids)} RELLIS sequences: {seq_ids}")

    if len(seq_ids) < 2:
        print("Need at least 2 sequences for cross-validation.")
        sys.exit(1)

    experiments = [e for e in EXPERIMENTS
                   if args.losses is None or e["name"] in args.losses]
    fold_indices = list(range(len(seq_ids)))
    if args.folds is not None:
        fold_indices = [i for i in args.folds if 0 <= i < len(seq_ids)]

    print(f"Experiments : {[e['name'] for e in experiments]}")
    print(f"Folds       : {fold_indices} (val sequences: {[seq_ids[i] for i in fold_indices]})")
    print(f"Epochs/run  : {cfg['training']['epochs']}")
    print(f"Total runs  : {len(experiments) * len(fold_indices)}")

    log_base  = Path(cfg["logging"]["log_dir"])
    save_base = Path(cfg["logging"]["save_dir"])
    out_dir   = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # results[fold_idx][exp_name] = metrics dict
    results: dict[int, dict[str, dict]] = {i: {} for i in fold_indices}

    for fold_idx in fold_indices:
        val_seq   = seq_ids[fold_idx]
        train_ids = [s for s in seq_ids if s != val_seq]
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx}  —  val: {val_seq}  |  train: {train_ids}")
        print(f"{'='*70}")

        for exp in experiments:
            print(f"\n  [{exp['name']}]  loss={exp['loss']['name']}")

            if args.dry_run:
                results[fold_idx][exp["name"]] = {"f1": 0.0, "iou": 0.0, "pr_auc": 0.0}
                continue

            log_dir  = str(log_base  / f"fold{fold_idx}_{val_seq}" / exp["name"])
            save_dir = str(save_base / f"fold{fold_idx}_{val_seq}" / exp["name"])
            os.makedirs(log_dir,  exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            try:
                metrics = run_one(exp, train_ids, [val_seq], cfg, log_dir, save_dir)
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
                )

    # ── Aggregate ────────────────────────────────────────────────────────────

    KEY = "f1"  # primary ranking metric

    agg: dict[str, dict] = {}
    for exp in experiments:
        vals = [results[i][exp["name"]].get(KEY, float("nan"))
                for i in fold_indices
                if results[i].get(exp["name"])]
        vals = [v for v in vals if not np.isnan(v)]
        agg[exp["name"]] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std":  float(np.std(vals))  if vals else float("nan"),
            "n":    len(vals),
        }

    # Rank matrix (lower rank = better)
    rank_matrix = np.zeros((len(fold_indices), len(experiments)))
    for fi, fold_idx in enumerate(fold_indices):
        fold_vals = [(exp["name"], results[fold_idx].get(exp["name"], {}).get(KEY, float("nan")))
                     for exp in experiments]
        sorted_names = [n for n, _ in sorted(fold_vals, key=lambda x: -x[1])]
        for ei, exp in enumerate(experiments):
            rank = sorted_names.index(exp["name"]) + 1 if exp["name"] in sorted_names else len(experiments)
            rank_matrix[fi, ei] = rank

    w = kendalls_w(rank_matrix)

    # ── Print summary table ───────────────────────────────────────────────────

    sorted_exps = sorted(experiments, key=lambda e: agg[e["name"]]["mean"], reverse=True)
    W = max(len(e["name"]) for e in experiments)

    print(f"\n{'='*80}")
    print(f"  CROSS-VALIDATION SUMMARY  ({len(fold_indices)} folds, metric={KEY})")
    print(f"  Kendall's W = {w:.3f}  {'(stable ✓)' if w > 0.7 else '(unstable — more data needed)'}")
    print(f"{'='*80}")
    fold_headers = "  ".join(f"fold{i:1d}" for i in fold_indices)
    print(f"{'Loss':<{W}}  {'Mean':>7}  {'Std':>6}  {fold_headers}")
    print("-" * 80)

    for exp in sorted_exps:
        name = exp["name"]
        m = agg[name]
        fold_vals_str = "  ".join(
            f"{results[i].get(name, {}).get(KEY, float('nan')):.3f}"
            for i in fold_indices
        )
        print(f"{name:<{W}}  {m['mean']:>7.4f}  {m['std']:>6.4f}  {fold_vals_str}")

    # ── Save results ─────────────────────────────────────────────────────────

    out_json = out_dir / "rellis_cv_results.json"
    out_csv  = out_dir / "rellis_cv_results.csv"

    full = {
        "kendalls_w": w,
        "folds":      {seq_ids[i]: {e["name"]: results[i].get(e["name"], {})
                                    for e in experiments}
                       for i in fold_indices},
        "aggregate":  agg,
    }
    with open(out_json, "w") as f:
        json.dump(full, f, indent=2)

    fields = ["loss", "mean_f1", "std_f1"] + [f"fold{i}_f1" for i in fold_indices]
    rows = []
    for exp in sorted_exps:
        row = {
            "loss":    exp["name"],
            "mean_f1": agg[exp["name"]]["mean"],
            "std_f1":  agg[exp["name"]]["std"],
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
    print(f"  W > 0.7  → rankings stable, results publishable as-is")
    print(f"  W > 0.5  → moderate agreement, consider adding TartanDrive")
    print(f"  W < 0.5  → high variance, rankings unreliable, more data needed")


if __name__ == "__main__":
    main()
