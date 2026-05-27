"""Train SparseTravNet on GOOSE-3D traversability (trajectory GT labels).

The model predicts per-voxel traversability using trav_gt (computed from
robot trajectory) as ground truth.  At validation time, agreement with the
terrain-based estimate (trav_terrain) is also reported as a secondary metric
to evaluate how well each loss function aligns geometric and trajectory priors.

Usage:
    python -m scripts.train_goose resources/train_goose.yaml
    python -m scripts.train_goose resources/train_goose.yaml loss.name=focal
    python -m scripts.train_goose resources/train_goose.yaml loss.name=nnpu loss.prior=0.4
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

sys.path.insert(0, str(Path(__file__).parents[1]))

from apairo import Goose3DDataset
from datasets.goose_trav import GooseTravTorchDataset, goose_trav_collate
from src.losses import TRAV_LOSSES
from src.models.sparse_trav_net import SparseTravNet


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_cfg(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class BinaryMetrics:
    def __init__(self) -> None:
        self.TP = self.FP = self.FN = self.TN = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> None:
        preds = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(bool)
        y = labels.cpu().numpy().astype(bool)
        self.TP += int(( preds &  y).sum())
        self.FP += int(( preds & ~y).sum())
        self.FN += int((~preds &  y).sum())
        self.TN += int((~preds & ~y).sum())

    def compute(self) -> dict[str, float]:
        prec = self.TP / (self.TP + self.FP + 1e-8)
        rec  = self.TP / (self.TP + self.FN + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        acc  = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN + 1e-8)
        return {"precision": prec, "recall": rec, "f1": f1, "acc": acc}


def terrain_agreement(logits: torch.Tensor, terrain: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of voxels where model prediction agrees with terrain estimate."""
    pred = (torch.sigmoid(logits) > threshold).cpu().numpy()
    terr = terrain.cpu().numpy().astype(bool)
    return float((pred.astype(bool) == terr).mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: dict) -> None:
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    dc = cfg["data"]
    train_ds = GooseTravTorchDataset(
        root_dir=dc["root"],
        split="train",
        voxel_size=dc["voxel_size"],
        max_rad=dc["max_rad"],
        min_pos=dc.get("min_pos", 1),
    )
    val_ds = GooseTravTorchDataset(
        root_dir=dc["root"],
        split="val",
        voxel_size=dc["voxel_size"],
        max_rad=dc["max_rad"],
        min_pos=dc.get("min_pos", 1),
    )

    tc = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc.get("num_workers", 4), pin_memory=True,
        collate_fn=goose_trav_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc.get("num_workers", 4), pin_memory=True,
        collate_fn=goose_trav_collate,
    )

    # For terrain agreement: load trav_terrain at validation time
    val_terrain_ds = Goose3DDataset(
        Path(dc["root"]),
        keys=["trav_terrain"],
        split="val",
    )

    print(f"Train: {len(train_ds)} scans / Val: {len(val_ds)} scans")

    mc = cfg["model"]
    model = SparseTravNet(in_channels=mc.get("in_channels", 4), cr=mc.get("cr", 1.0)).to(device)

    lc = cfg["loss"]
    criterion = TRAV_LOSSES[lc["name"]](lc).to(device)
    print(f"Loss: {lc['name']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tc.get("lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["epochs"], eta_min=1e-5
    )

    logc = cfg["logging"]
    exp_name = logc.get("exp_name", lc["name"])
    log_dir  = os.path.join(logc["log_dir"],  exp_name)
    save_dir = os.path.join(logc["save_dir"], exp_name)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_f1 = 0.0
    scaler  = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, tc["epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_losses, train_m = [], BinaryMetrics()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_m.update(logits.detach(), labels)
            print(
                f"\rEpoch {epoch}/{tc['epochs']} [{i+1}/{len(train_loader)}]"
                f"  loss={np.mean(train_losses):.4f}", end=""
            )

        print()
        scheduler.step()

        tm = train_m.compute()
        tm["loss"] = float(np.mean(train_losses))
        for k, v in tm.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        # ── val ────────────────────────────────────────────────────────────
        model.eval()
        val_losses, val_m = [], BinaryMetrics()
        terrain_agreements: list[float] = []

        with torch.no_grad():
            for scan_idx, batch in enumerate(val_loader):
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
                val_m.update(logits, labels)

                # Terrain agreement — per-voxel (approximate: uses raw trav_terrain)
                if scan_idx < len(val_terrain_ds):
                    terrain_raw = np.asarray(val_terrain_ds[scan_idx].data["trav_terrain"])
                    if len(terrain_raw) == logits.shape[0]:
                        terr = torch.from_numpy(terrain_raw > 0.5)
                        terrain_agreements.append(terrain_agreement(logits.cpu(), terr))

        vm = val_m.compute()
        vm["loss"]             = float(np.mean(val_losses))
        vm["terrain_agreement"] = float(np.mean(terrain_agreements)) if terrain_agreements else 0.0

        for k, v in vm.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        print(
            f"  val  loss={vm['loss']:.4f}  f1={vm['f1']:.4f}"
            f"  terrain_agree={vm['terrain_agreement']:.4f}"
        )

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            ckpt = os.path.join(save_dir, "best.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → best saved (f1={best_f1:.4f})")

    writer.close()
    print(f"\nDone. Best val F1={best_f1:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args()
    cfg = load_cfg(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
