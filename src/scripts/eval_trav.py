"""Evaluate a trained SparseTravNet checkpoint on the val set.

Usage:
    # Evaluate one checkpoint
    python -m src.scripts.eval_trav data/checkpoints/trav/bce/best.pth resources/train_trav.yaml

    # Evaluate all checkpoints from a study run
    python -m src.scripts.eval_trav --all resources/train_trav.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets.tartan_trav_train import TartanTravTrainDataset, trav_collate
from src.models.sparse_trav_net import SparseTravNet


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    preds = (torch.sigmoid(logits) > threshold).numpy()
    y = labels.numpy()
    TP = int(((preds == 1) & (y == 1)).sum())
    FP = int(((preds == 1) & (y == 0)).sum())
    FN = int(((preds == 0) & (y == 1)).sum())
    TN = int(((preds == 0) & (y == 0)).sum())
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (TP + TN) / (TP + FP + FN + TN + 1e-8)
    pos_ratio = y.mean()
    pred_ratio = preds.mean()
    return {
        "f1": f1, "precision": precision, "recall": recall, "acc": acc,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "pos_ratio": float(pos_ratio), "pred_ratio": float(pred_ratio),
    }


def evaluate(ckpt_path: Path, cfg: dict, device: torch.device) -> dict:
    ds_cfg = cfg["data"]
    val_ds = TartanTravTrainDataset(
        tartan_root=ds_cfg["root"],
        voxel_size=ds_cfg["voxel_size"],
        max_rad=ds_cfg["max_rad"],
        train_frac=ds_cfg.get("train_frac", 0.8),
        max_sequences=ds_cfg.get("max_sequences"),
        split="val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=trav_collate,
    )

    m_cfg = cfg["model"]
    model = SparseTravNet(
        in_channels=m_cfg.get("in_channels", 4),
        cr=m_cfg.get("cr", 1.0),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            st     = batch["sparse_input"].to(device)
            labels = batch["labels"].to(device)
            if st.feats.shape[0] == 0:
                continue
            try:
                logits = model(st)
            except Exception:
                continue
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    if not all_logits:
        return {"error": "no valid batches"}

    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["n_scans"] = len(val_ds)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", help="Path to .pth checkpoint")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints under save_dir")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all:
        save_dir = Path(cfg["logging"]["save_dir"])
        ckpts = sorted(save_dir.glob("*/best.pth"))
        if not ckpts:
            print(f"No checkpoints found in {save_dir}")
            sys.exit(1)
    else:
        if not args.checkpoint:
            parser.error("Provide a checkpoint path or use --all")
        ckpts = [Path(args.checkpoint)]

    results = []
    print(f"\n{'Loss':<12} {'F1':>7} {'Precision':>10} {'Recall':>8} {'Acc':>7} {'PredRatio':>10}")
    print("-" * 60)

    for ckpt in ckpts:
        name = ckpt.parent.name
        print(f"Evaluating {name} ...", end="\r")
        metrics = evaluate(ckpt, cfg, device)
        results.append({"loss": name, "checkpoint": str(ckpt), **metrics})

        if "error" in metrics:
            print(f"{name:<12}  ERROR: {metrics['error']}")
        else:
            print(
                f"{name:<12} {metrics['f1']:>7.4f} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>8.4f} {metrics['acc']:>7.4f} {metrics['pred_ratio']:>10.4f}"
            )

    # Save
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    csv_fields = ["loss", "f1", "precision", "recall", "acc", "pos_ratio", "pred_ratio", "TP", "FP", "FN", "TN", "n_scans", "checkpoint"]
    with open(out_dir / "eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to data/eval_results.json and data/eval_results.csv")


if __name__ == "__main__":
    main()
