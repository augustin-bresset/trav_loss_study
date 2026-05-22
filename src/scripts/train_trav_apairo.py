"""Train a SparseTravNet on TartanDrive with a configurable binary loss.

Uses the Apairo-backed dataset (TartanTravTorchDataset).
Sequences must have been preprocessed with preprocess_trav.py first.

Usage:
    python -m src.scripts.train_trav_apairo resources/train_trav.yaml
    python -m src.scripts.train_trav_apairo resources/train_trav.yaml loss.name=focal
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets.tartan_trav_torch import TartanTravTorchDataset, trav_collate_apairo
from src.losses import TRAV_LOSSES
from src.models.sparse_trav_net import SparseTravNet


# ── helpers ───────────────────────────────────────────────────────────────────

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


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    preds = (torch.sigmoid(logits) > threshold).cpu().numpy()
    y = labels.cpu().numpy()
    TP = int(((preds == 1) & (y == 1)).sum())
    FP = int(((preds == 1) & (y == 0)).sum())
    FN = int(((preds == 0) & (y == 1)).sum())
    TN = int(((preds == 0) & (y == 0)).sum())
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (TP + TN) / (TP + FP + FN + TN + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "acc": acc}


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    # ── datasets ──────────────────────────────────────────────────────────────
    ds_cfg = cfg["data"]
    ds_kwargs = dict(
        tartan_root=ds_cfg["root"],
        voxel_size=ds_cfg["voxel_size"],
        max_rad=ds_cfg["max_rad"],
        train_frac=ds_cfg.get("train_frac", 0.8),
        max_sequences=ds_cfg.get("max_sequences"),
    )
    train_ds = TartanTravTorchDataset(split="train", **ds_kwargs)
    val_ds   = TartanTravTorchDataset(split="val",   **ds_kwargs)

    t_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=t_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=trav_collate_apairo,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=trav_collate_apairo,
    )

    print(f"Train: {len(train_ds)} scans / Val: {len(val_ds)} scans")

    # ── model ─────────────────────────────────────────────────────────────────
    m_cfg = cfg["model"]
    model = SparseTravNet(
        in_channels=m_cfg.get("in_channels", 4),
        cr=m_cfg.get("cr", 1.0),
    ).to(device)

    # ── loss ──────────────────────────────────────────────────────────────────
    l_cfg = cfg["loss"]
    loss_name = l_cfg["name"]
    criterion = TRAV_LOSSES[loss_name](l_cfg).to(device)
    print(f"Loss: {loss_name}  |  {criterion}")

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"], eta_min=1e-5
    )

    # ── logging ───────────────────────────────────────────────────────────────
    log_cfg = cfg["logging"]
    exp_name = log_cfg.get("exp_name", loss_name)
    log_dir  = os.path.join(log_cfg["log_dir"],  exp_name)
    save_dir = os.path.join(log_cfg["save_dir"], exp_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_f1   = 0.0
    best_ckpt = os.path.join(save_dir, "best.pth")
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, t_cfg["epochs"] + 1):
        # ── train ──
        model.train()
        train_losses: list[float] = []
        all_logits, all_labels = [], []

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
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

            print(
                f"\rEpoch {epoch}/{t_cfg['epochs']}  "
                f"[{i+1}/{len(train_loader)}]  "
                f"loss={np.mean(train_losses):.4f}",
                end="",
            )

        print()
        scheduler.step()

        train_logits  = torch.cat(all_logits)
        train_labels  = torch.cat(all_labels)
        train_metrics = binary_metrics(train_logits, train_labels)
        train_metrics["loss"] = float(np.mean(train_losses))

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        # ── val ──
        model.eval()
        val_losses: list[float] = []
        val_logits_all, val_labels_all = [], []

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
                except IndexError:
                    continue
                val_losses.append(loss.item())
                val_logits_all.append(logits.cpu())
                val_labels_all.append(labels.cpu())

        val_logits  = torch.cat(val_logits_all)
        val_labels  = torch.cat(val_labels_all)
        val_metrics = binary_metrics(val_logits, val_labels)
        val_metrics["loss"] = float(np.mean(val_losses))

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


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
