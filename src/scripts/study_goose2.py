"""Extended loss/hyperparameter study on GOOSE-3D — second round.

Motivated by study_goose results (data/goose_study_results.json):
  - BCE / focal / uPU :  precision≈0.87 but recall≤0.23  (too conservative)
  - nnPU (prior=0.5)  :  F1=0.769 but recall=1.0         (too aggressive, prior too high)

Strategy:
  1. Sweep nnPU prior downward (0.2 / 0.3 / 0.4) to find the
     precision-recall sweet spot.
  2. Add a non-zero beta floor to nnPU to stabilise negative-risk clamping.
  3. Push focal harder (gamma=3-5, pos_weight=6-8) to rescue recall.
  4. Three new losses: Tversky (FN-weighted Dice), Lovász-Hinge (IoU proxy),
     FocalnnPU (nnPU with focal base), ASL (asymmetric focusing).

Outputs:
  data/goose_study2_results.json
  data/goose_study2_results.csv

Usage:
    python -m src.scripts.study_goose2
    python -m src.scripts.study_goose2 --losses nnpu_prior30 lovasz
    python -m src.scripts.study_goose2 --config resources/train_goose.yaml
    python -m src.scripts.study_goose2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
# Each entry:  name (str), overrides (dict[str, scalar])
#
# Group A — nnPU prior sweep
#   Goal: prior=0.5 was too high (recall=1, model always predicts positive).
#         Lower prior → less pressure to predict positive → better precision.
#
# Group B — nnPU beta (floor) sweep
#   Goal: beta>0 cuts the gradient when neg_risk undershoots, which may help
#         precision without sacrificing too much recall.
#
# Group C — focal parameter sweep
#   Goal: gamma=2 / pw=3.6 left recall at 0.21. Push both up.
#
# Group D — new losses
#   Tversky    : Dice with alpha=0.3 (FP) / beta=0.7 (FN) → recall-oriented
#   Lovász     : directly optimises binary IoU
#   FocalnnPU  : nnPU + focal base (hard-example focus in PU setting)
#   ASL        : asymmetric loss, clips easy negatives to zero gradient

EXPERIMENTS: list[dict] = [
    # ── Group A: nnPU prior sweep ─────────────────────────────────────────
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
    # ── Group B: nnPU beta sweep (prior fixed at 0.5 — same as study 1) ──
    {
        "name": "nnpu_beta01",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.50, "loss.beta": 0.01},
    },
    {
        "name": "nnpu_beta05",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.50, "loss.beta": 0.05},
    },
    # Best-guess combined: lower prior + small floor
    {
        "name": "nnpu_prior30_beta02",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.30, "loss.beta": 0.02},
    },
    # ── Group C: focal parameter sweep ───────────────────────────────────
    {
        "name": "focal_g3_pw6",
        "overrides": {"loss.name": "focal", "loss.gamma": 3.0, "loss.pos_weight": 6.0},
    },
    {
        "name": "focal_g5_pw8",
        "overrides": {"loss.name": "focal", "loss.gamma": 5.0, "loss.pos_weight": 8.0},
    },
    # BCE with larger pos_weight (no gamma term — isolates the effect)
    {
        "name": "bce_pw6",
        "overrides": {"loss.name": "bce", "loss.pos_weight": 6.0},
    },
    # ── Group D: new losses ───────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_cmd(config: str, name: str, overrides: dict) -> list[str]:
    cmd = [sys.executable, "-m", "src.scripts.train_goose", config]
    cmd += [f"logging.exp_name={name}"]
    cmd += [f"{k}={v}" for k, v in overrides.items()]
    return cmd


def run_experiment(cmd: list[str]) -> int:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


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


def summarise(scalars: dict[str, list[float]]) -> dict[str, float | None]:
    def _best(tag, fn):
        vals = scalars.get(tag)
        return fn(vals) if vals else None

    return {
        "best_val_f1":        _best("val/f1",        max),
        "best_val_precision": _best("val/precision",  max),
        "best_val_recall":    _best("val/recall",     max),
        "best_val_loss":      _best("val/loss",       min),
        "final_train_loss":   scalars.get("train/loss", [None])[-1],
        "n_epochs_logged":    len(scalars.get("val/f1", [])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",  default="resources/train_goose.yaml")
    parser.add_argument("--losses",  nargs="+", help="Subset of experiment names to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    log_dir_base = Path(base_cfg["logging"]["log_dir"])
    out_dir      = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "goose_study2_results.json"
    results_csv  = out_dir / "goose_study2_results.csv"

    experiments = [
        e for e in EXPERIMENTS
        if args.losses is None or e["name"] in args.losses
    ]
    if not experiments:
        print("No matching experiments. Available:", [e["name"] for e in EXPERIMENTS])
        sys.exit(1)

    records: list[dict] = []

    for exp in experiments:
        name, overrides = exp["name"], exp["overrides"]
        print(f"\n{'='*60}")
        print(f"  Experiment: {name}")
        print(f"{'='*60}")

        cmd = build_cmd(args.config, name, overrides)

        if args.dry_run:
            print("  [dry-run] would run:", " ".join(cmd))
            records.append({"loss": name, "status": "dry-run"})
            continue

        rc     = run_experiment(cmd)
        status = "ok" if rc == 0 else f"error({rc})"

        scalars = read_tb_scalars(log_dir_base / name)
        summary = summarise(scalars)
        record  = {"loss": name, "status": status, **summary, "overrides": json.dumps(overrides)}
        records.append(record)

        f1_str = f"{summary['best_val_f1']:.4f}" if summary["best_val_f1"] is not None else "N/A"
        print(f"  → status={status}  best_val_f1={f1_str}")

    with open(results_json, "w") as f:
        json.dump(records, f, indent=2)

    csv_fields = [
        "loss", "status",
        "best_val_f1", "best_val_precision", "best_val_recall",
        "best_val_loss", "final_train_loss", "n_epochs_logged",
        "overrides",
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  GOOSE STUDY 2 SUMMARY")
    print(f"{'='*72}")
    print(f"{'Loss':<22}  {'Val F1':>7}  {'Prec':>7}  {'Recall':>7}  {'Val Loss':>9}  Status")
    print("-" * 72)

    # Sort by best_val_f1 descending for readability
    sorted_records = sorted(
        [r for r in records if r.get("best_val_f1") is not None],
        key=lambda r: r["best_val_f1"],
        reverse=True,
    ) + [r for r in records if r.get("best_val_f1") is None]

    for r in sorted_records:
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "N/A"
        print(
            f"{r['loss']:<22}  "
            f"{fmt(r.get('best_val_f1')):>7}  "
            f"{fmt(r.get('best_val_precision')):>7}  "
            f"{fmt(r.get('best_val_recall')):>7}  "
            f"{fmt(r.get('best_val_loss')):>9}  "
            f"{r['status']}"
        )
    print(f"\nResults saved to {results_json} and {results_csv}")


if __name__ == "__main__":
    main()
