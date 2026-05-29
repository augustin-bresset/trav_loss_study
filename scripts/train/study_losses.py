"""Compare multiple loss functions on GOOSE-3D traversability.

Each loss is trained for the same number of epochs.  Results are evaluated on:
  - Val F1 / precision / recall  (w.r.t. trajectory GT, ``trav_gt``)
  - Terrain agreement            (fraction of voxels where model agrees with
                                  ``trav_terrain`` — geometric prior, not GT)

Results are saved to ``data/study_results.{json,csv}`` and printed as a table.

Experiment grid is motivated by study_goose results (data/goose_study_results.json):
  - BCE / focal / uPU had precision≈0.87 but recall≤0.23 (too conservative)
  - nnPU (prior=0.5) had recall=1.0 (too aggressive, predicts everything positive)
  → Group A: sweep nnPU prior downward to find the precision-recall sweet spot
  → Group B: nnPU beta floor to stabilise gradient clamping
  → Group C: push focal harder (gamma, pos_weight) to rescue recall
  → Group D: new losses (Tversky, Lovász, FocalnnPU, ASL)

Usage:
    python -m scripts.study_losses                              # all losses
    python -m scripts.study_losses --losses bce nnpu_prior30   # subset
    python -m scripts.study_losses --config resources/train_goose_trav.yaml
    python -m scripts.study_losses --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

EXPERIMENTS: list[dict] = [
    # ── Baselines (study 1 reference) ────────────────────────────────────────
    {
        "name": "bce",
        "overrides": {"loss.name": "bce"},
    },
    {
        "name": "focal",
        "overrides": {"loss.name": "focal", "loss.gamma": 2.0, "loss.pos_weight": 3.6},
    },
    # ── Group A: nnPU prior sweep ─────────────────────────────────────────────
    # study 1: nnPU(prior=0.5) → recall=1.0, model predicts everything positive
    # → lower prior reduces pressure toward positive, improves precision
    {
        "name": "nnpu_prior20",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.20, "loss.beta": 0.0},
    },
    {
        "name": "nnpu_prior30",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.30, "loss.beta": 0.0},
    },
    {
        "name": "nnpu_prior40",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.40, "loss.beta": 0.0},
    },
    # ── Group B: nnPU beta floor sweep (prior fixed at 0.5) ───────────────────
    # beta > 0 clamps the neg-risk floor → gradient cut-off is less aggressive
    {
        "name": "nnpu_beta01",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.50, "loss.beta": 0.01},
    },
    {
        "name": "nnpu_beta05",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.50, "loss.beta": 0.05},
    },
    # best-guess combined: lower prior + small floor
    {
        "name": "nnpu_p30_b02",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.30, "loss.beta": 0.02},
    },
    # ── Group C: focal parameter sweep ───────────────────────────────────────
    # study 1: focal(gamma=2, pw=3.6) → recall=0.21 still too low
    # → push gamma and pos_weight harder to force recall up
    {
        "name": "focal_g3_pw6",
        "overrides": {"loss.name": "focal", "loss.gamma": 3.0, "loss.pos_weight": 6.0},
    },
    {
        "name": "focal_g5_pw8",
        "overrides": {"loss.name": "focal", "loss.gamma": 5.0, "loss.pos_weight": 8.0},
    },
    {
        "name": "bce_pw6",
        "overrides": {"loss.name": "bce", "loss.pos_weight": 6.0},
    },
    # ── Group D: new losses ───────────────────────────────────────────────────
    {
        "name": "tversky_a03",
        "overrides": {
            "loss.name": "tversky",
            "loss.tversky_alpha": 0.3,
            "loss.tversky_beta": 0.7,
        },
    },
    {
        "name": "tversky_a02",
        "overrides": {
            "loss.name": "tversky",
            "loss.tversky_alpha": 0.2,
            "loss.tversky_beta": 0.8,
        },
    },
    {
        "name": "lovasz",
        "overrides": {"loss.name": "lovasz"},
    },
    {
        "name": "focal_nnpu_p30",
        "overrides": {
            "loss.name": "focal_nnpu",
            "loss.prior": 0.30,
            "loss.gamma": 2.0,
            "loss.beta": 0.0,
        },
    },
    {
        "name": "focal_nnpu_p40",
        "overrides": {
            "loss.name": "focal_nnpu",
            "loss.prior": 0.40,
            "loss.gamma": 2.0,
            "loss.beta": 0.0,
        },
    },
    {
        "name": "asl",
        "overrides": {
            "loss.name": "asl",
            "loss.gamma_pos": 0.0,
            "loss.gamma_neg": 4.0,
            "loss.asl_clip": 0.05,
        },
    },
]


def build_cmd(config: str, exp: dict) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.train_goose", config]
    cmd += [f"logging.exp_name={exp['name']}"]
    cmd += [f"{k}={v}" for k, v in exp["overrides"].items()]
    return cmd


def read_tb_scalars(log_dir: Path) -> dict[str, list[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    if not log_dir.exists():
        return {}
    try:
        ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        ea.Reload()
        return {tag: [e.value for e in ea.Scalars(tag)] for tag in ea.Tags().get("scalars", [])}
    except Exception:
        return {}


def summarise(scalars: dict[str, list[float]]) -> dict:
    def _best(tag, fn):
        vals = scalars.get(tag)
        return fn(vals) if vals else None

    return {
        "best_val_f1":             _best("val/f1",                 max),
        "best_val_precision":      _best("val/precision",           max),
        "best_val_recall":         _best("val/recall",              max),
        "best_val_loss":           _best("val/loss",                min),
        "best_alt_agreement":  _best("val/alt_agreement",   max),
        "final_train_loss":        (scalars.get("train/loss") or [None])[-1],
        "n_epochs":                len(scalars.get("val/f1") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="resources/train_goose_trav.yaml")
    parser.add_argument("--losses",  nargs="+", help="Subset of loss names to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    log_dir_base = Path(base_cfg["logging"]["log_dir"])
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "study_results.json"
    results_csv  = out_dir / "study_results.csv"

    experiments = [e for e in EXPERIMENTS if args.losses is None or e["name"] in args.losses]
    if not experiments:
        print("No matching experiments. Available:", [e["name"] for e in EXPERIMENTS])
        sys.exit(1)

    records: list[dict] = []

    for exp in experiments:
        print(f"\n{'='*64}")
        print(f"  Experiment: {exp['name']}")
        print(f"{'='*64}")

        cmd = build_cmd(args.config, exp)
        if args.dry_run:
            print("  [dry-run]", " ".join(cmd))
            records.append({"loss": exp["name"], "status": "dry-run"})
            continue

        print("  $", " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        status = "ok" if rc == 0 else f"error({rc})"

        scalars = read_tb_scalars(log_dir_base / exp["name"])
        summary = summarise(scalars)
        records.append({"loss": exp["name"], "status": status,
                        **summary, "overrides": json.dumps(exp["overrides"])})

        def _fmt(v): return f"{v:.4f}" if isinstance(v, float) else "N/A"
        print(
            f"  → f1={_fmt(summary['best_val_f1'])}"
            f"  terrain={_fmt(summary['best_alt_agreement'])}"
            f"  status={status}"
        )

    # ── save results ──────────────────────────────────────────────────────────
    with open(results_json, "w") as f:
        json.dump(records, f, indent=2)

    fields = [
        "loss", "status",
        "best_val_f1", "best_val_precision", "best_val_recall",
        "best_val_loss", "best_alt_agreement",
        "final_train_loss", "n_epochs", "overrides",
    ]
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # ── summary table (sorted by best_val_f1 desc) ───────────────────────────
    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "  N/A"
    sorted_records = sorted(
        [r for r in records if r.get("best_val_f1") is not None],
        key=lambda r: r["best_val_f1"],
        reverse=True,
    ) + [r for r in records if r.get("best_val_f1") is None]

    W = max(len(r["loss"]) for r in records)
    print(f"\n{'='*80}")
    print("  STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Loss':<{W}}  {'Val F1':>7}  {'Prec':>7}  {'Recall':>7}  {'Terrain↑':>9}  Status")
    print("-" * 80)
    for r in sorted_records:
        print(
            f"{r['loss']:<{W}}  "
            f"{fmt(r.get('best_val_f1')):>7}  "
            f"{fmt(r.get('best_val_precision')):>7}  "
            f"{fmt(r.get('best_val_recall')):>7}  "
            f"{fmt(r.get('best_alt_agreement')):>9}  "
            f"{r['status']}"
        )
    print(f"\nSaved → {results_json}  /  {results_csv}")


if __name__ == "__main__":
    main()
