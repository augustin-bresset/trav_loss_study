"""Compare multiple loss functions on GOOSE-3D traversability.

Each loss is trained for the same number of epochs.  Results are evaluated on:
  - Val F1 / precision / recall  (w.r.t. trajectory GT, ``trav_gt``)
  - Terrain agreement            (fraction of predictions matching ``trav_terrain``)

Results are saved to ``data/study_results.{json,csv}`` and printed as a table.

Usage:
    python -m scripts.study_losses                              # all losses
    python -m scripts.study_losses --losses bce nnpu           # subset
    python -m scripts.study_losses --config resources/train_goose.yaml
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
    {
        "name": "bce",
        "overrides": {"loss.name": "bce"},
    },
    {
        "name": "focal",
        "overrides": {"loss.name": "focal", "loss.gamma": 2.0, "loss.pos_weight": 3.6},
    },
    {
        "name": "upu",
        "overrides": {"loss.name": "upu", "loss.prior": 0.50},
    },
    {
        "name": "nnpu",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.50, "loss.beta": 0.0},
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
        "best_terrain_agreement":  _best("val/terrain_agreement",   max),
        "final_train_loss":        (scalars.get("train/loss") or [None])[-1],
        "n_epochs":                len(scalars.get("val/f1") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="resources/train_goose.yaml")
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
            f"  terrain={_fmt(summary['best_terrain_agreement'])}"
            f"  status={status}"
        )

    # ── save results ──────────────────────────────────────────────────────────
    with open(results_json, "w") as f:
        json.dump(records, f, indent=2)

    fields = [
        "loss", "status",
        "best_val_f1", "best_val_precision", "best_val_recall",
        "best_val_loss", "best_terrain_agreement",
        "final_train_loss", "n_epochs", "overrides",
    ]
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  STUDY SUMMARY")
    print(f"{'='*72}")
    print(f"{'Loss':<8}  {'Val F1':>7}  {'Precision':>9}  {'Recall':>7}  {'Terrain↑':>9}  Status")
    print("-" * 72)
    for r in records:
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "N/A"
        print(
            f"{r['loss']:<8}  {fmt(r.get('best_val_f1')):>7}  "
            f"{fmt(r.get('best_val_precision')):>9}  "
            f"{fmt(r.get('best_val_recall')):>7}  "
            f"{fmt(r.get('best_terrain_agreement')):>9}  "
            f"{r['status']}"
        )
    print(f"\nSaved → {results_json}  /  {results_csv}")


if __name__ == "__main__":
    main()
