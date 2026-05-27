"""Train a SparseTravNet on GOOSE-3D with a configurable binary loss.

The problem is framed as Positive-Unlabeled (PU) learning:
  label=1  →  confirmed traversable  (positive)
  label=0  →  unlabeled              (not necessarily non-traversable)

Only scans containing at least one positive-labeled point are used,
so every batch is guaranteed to carry a P set for uPU / nnPU losses.

Usage:
    python -m src.scripts.train_goose resources/train_goose.yaml
    python -m src.scripts.train_goose resources/train_goose.yaml loss.name=nnpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets.goose_trav import GooseTravDataset, goose_trav_collate
from src.losses import TRAV_LOSSES
from src.models.sparse_trav_net import SparseTravNet


def load_cfg(yaml_path: str, cli_overrides: list[str]) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    for ov in cli_overrides:
        key_path, _, value = ov.partition("=")
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        node[keys[-1]] = value
    return cfg


class RunningMetrics:
    """Accumulate TP/FP/FN/TN without storing all predictions."""

    def __init__(self) -> None:
        self.TP = self.FP = self.FN = self.TN = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> None:
        preds = (torch.sigmoid(logits) > threshold)
        y = labels.bool()
        self.TP += int((preds & y).sum())
        self.FP += int((preds & ~y).sum())
        self.FN += int((~preds & y).sum())
        self.TN += int((~preds & ~y).sum())

    def compute(self) -> dict:
        p = self.TP / (self.TP + self.FP + 1e-8)
        r = self.TP / (self.TP + self.FN + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        acc = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN + 1e-8)
        return {"precision": p, "recall": r, "f1": f1, "acc": acc}


def train(cfg: dict) -> None:
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    ds_cfg = cfg["data"]
    ds_kwargs = dict(
        voxel_size=ds_cfg["voxel_size"],
        max_rad=ds_cfg["max_rad"],
        min_pos=ds_cfg.get("min_pos", 1),
    )
    train_ds = GooseTravDataset(root_dir=ds_cfg["train_dir"], **ds_kwargs)
    val_ds   = GooseTravDataset(root_dir=ds_cfg["val_dir"],   **ds_kwargs)

    t_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=t_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=goose_trav_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=goose_trav_collate,
    )

    print(f"Train: {len(train_ds)} scans / Val: {len(val_ds)} scans")

    m_cfg = cfg["model"]
    model = SparseTravNet(
        in_channels=m_cfg.get("in_channels", 4),
        cr=m_cfg.get("cr", 1.0),
    ).to(device)

    l_cfg = cfg["loss"]
    loss_name = l_cfg["name"]
    criterion = TRAV_LOSSES[loss_name](l_cfg).to(device)
    print(f"Loss: {loss_name}  |  {criterion}")

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=1e-5
    )

    log_cfg  = cfg["logging"]
    exp_name = log_cfg.get("exp_name", loss_name)
    log_dir  = os.path.join(log_cfg["log_dir"],  exp_name)
    save_dir = os.path.join(log_cfg["save_dir"], exp_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_f1   = 0.0
    best_ckpt = os.path.join(save_dir, "best.pth")
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        train_losses: list[float] = []
        train_acc = RunningMetrics()

        for i, batch in enumerate(train_loader):
            st     = batch["sparse_input"].to(device)
            labels = batch["labels"].to(device)

            if st.feats.shape[0] == 0:
                continue

            optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(st)
                    loss   = criterion(logits, labels.float())
            except Exception:
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_acc.update(logits.detach(), labels)

            print(
                f"\rEpoch {epoch}/{t_cfg['epochs']}  "
                f"[{i+1}/{len(train_loader)}]  "
                f"loss={np.mean(train_losses):.4f}",
                end="",
            )

        print()
        scheduler.step()

        train_metrics = train_acc.compute()
        train_metrics["loss"] = float(np.mean(train_losses))
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        model.eval()
        val_losses: list[float] = []
        val_acc = RunningMetrics()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in val_loader:
                st     = batch["sparse_input"].to(device)
                labels = batch["labels"].to(device)
                if st.feats.shape[0] == 0:
                    continue
                try:
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logits = model(st)
                        loss   = criterion(logits, labels.float())
                except Exception:
                    continue
                val_losses.append(loss.item())
                val_acc.update(logits, labels)

        val_metrics = val_acc.compute()
        val_metrics["loss"] = float(np.mean(val_losses)) if val_losses else float("nan")
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        print(
            f"  val  loss={val_metrics['loss']:.4f}  "
            f"f1={val_metrics['f1']:.4f}  "
            f"prec={val_metrics['precision']:.4f}  "
            f"rec={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  → best model saved (f1={best_f1:.4f})")

    writer.close()
    print(f"\nDone. Best val F1={best_f1:.4f}  checkpoint: {best_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
