"""Launch one training run per loss and aggregate results into a CSV summary.

Each experiment runs as a subprocess (clean GPU memory between runs).
Results are read from TensorBoard event files and saved to data/study_results.csv.

Usage:
    python -m src.scripts.study_losses                          # all losses
    python -m src.scripts.study_losses --losses bce focal       # subset
    python -m src.scripts.study_losses --config resources/train_trav.yaml
    python -m src.scripts.study_losses --dry-run                # print commands only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

# ── experiment definitions ────────────────────────────────────────────────────

EXPERIMENTS: list[dict] = [
    {
        "name":      "bce",
        "overrides": {"loss.name": "bce"},
    },
    {
        "name":      "focal",
        "overrides": {"loss.name": "focal", "loss.gamma": 2.0, "loss.pos_weight": 5.0},
    },
    {
        "name":      "upu",
        "overrides": {"loss.name": "upu", "loss.prior": 0.06},
    },
    {
        "name":      "nnpu",
        "overrides": {"loss.name": "nnpu", "loss.prior": 0.06, "loss.beta": 0.0},
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def build_cmd(config: str, name: str, overrides: dict) -> list[str]:
    cmd = [sys.executable, "-m", "src.scripts.train_trav", config]
    cmd += [f"logging.exp_name={name}"]
    cmd += [f"{k}={v}" for k, v in overrides.items()]
    return cmd


def run_experiment(cmd: list[str]) -> int:
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def read_tb_scalars(log_dir: Path) -> dict[str, list[float]]:
    """Return {tag: [values ordered by step]} from a TensorBoard event dir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return {
        tag: [e.value for e in ea.Scalars(tag)]
        for tag in ea.Tags().get("scalars", [])
    }


def summarise(scalars: dict[str, list[float]]) -> dict[str, float | None]:
    def _best(tag: str, fn) -> float | None:
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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="resources/train_trav.yaml")
    parser.add_argument("--losses",  nargs="+", help="Subset of loss names to run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    log_dir_base  = Path(base_cfg["logging"]["log_dir"])
    out_dir       = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json  = out_dir / "study_results.json"
    results_csv   = out_dir / "study_results.csv"

    experiments = [e for e in EXPERIMENTS if args.losses is None or e["name"] in args.losses]

    if not experiments:
        print("No matching experiments. Available:", [e["name"] for e in EXPERIMENTS])
        sys.exit(1)

    records: list[dict] = []

    for exp in experiments:
        name      = exp["name"]
        overrides = exp["overrides"]
        print(f"\n{'='*60}")
        print(f"  Experiment: {name}")
        print(f"{'='*60}")

        cmd = build_cmd(args.config, name, overrides)

        if args.dry_run:
            print("  [dry-run] would run:", " ".join(cmd))
            records.append({"loss": name, "status": "dry-run"})
            continue

        rc = run_experiment(cmd)
        status = "ok" if rc == 0 else f"error({rc})"

        scalars = read_tb_scalars(log_dir_base / name)
        summary = summarise(scalars)

        record = {"loss": name, "status": status, **summary, "overrides": json.dumps(overrides)}
        records.append(record)

        # print per-run summary immediately so partial results are visible
        f1 = summary["best_val_f1"]
        print(f"  → status={status}  best_val_f1={f1:.4f if f1 is not None else 'N/A'}")

    # ── persist ───────────────────────────────────────────────────────────────
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

    # ── final table ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Loss':<8}  {'Val F1':>7}  {'Precision':>9}  {'Recall':>7}  {'Val Loss':>9}  Status")
    print("-" * 60)
    for r in records:
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "N/A"
        print(
            f"{r['loss']:<8}  "
            f"{fmt(r.get('best_val_f1')):>7}  "
            f"{fmt(r.get('best_val_precision')):>9}  "
            f"{fmt(r.get('best_val_recall')):>7}  "
            f"{fmt(r.get('best_val_loss')):>9}  "
            f"{r['status']}"
        )

    print(f"\nResults saved to {results_json} and {results_csv}")


if __name__ == "__main__":
    main()
