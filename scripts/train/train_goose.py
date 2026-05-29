"""Train SparseTravNet on GOOSE-3D traversability (trajectory GT labels).

Usage:
    python -m scripts.train.train_goose resources/train_goose.yaml
    python -m scripts.train.train_goose resources/train_goose.yaml loss.name=focal
    python -m scripts.train.train_goose resources/train_goose.yaml loss.name=nnpu loss.prior=0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.datasets import GooseTorchDataset, sparse_collate
from src.losses import TRAV_LOSSES
from src.models.sparse_trav_net import SparseTravNet
from src.trainer import Trainer


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args()
    cfg = load_cfg(args.config, args.overrides)

    dc = cfg["data"]
    tc = cfg["training"]
    lc = cfg["loss"]
    logc = cfg["logging"]

    train_ds = GooseTorchDataset(
        root_dir=dc["root"],
        split="train",
        voxel_size=dc["voxel_size"],
        max_rad=dc["max_rad"],
        min_pos=dc.get("min_pos", 1),
    )
    val_ds = GooseTorchDataset(
        root_dir=dc["root"],
        split="val",
        voxel_size=dc["voxel_size"],
        max_rad=dc["max_rad"],
        min_pos=dc.get("min_pos", 1),
    )
    print(f"Train: {len(train_ds)} scans / Val: {len(val_ds)} scans")

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
    mc = cfg["model"]
    model = SparseTravNet(in_channels=mc.get("in_channels", 4), cr=mc.get("cr", 1.0)).to(device)

    criterion = TRAV_LOSSES[lc["name"]](lc).to(device)
    print(f"Loss: {lc['name']}")

    exp_name = logc.get("exp_name", lc["name"])
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=tc["epochs"],
        lr=tc.get("lr", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-4),
        log_dir=os.path.join(logc["log_dir"],  exp_name),
        save_dir=os.path.join(logc["save_dir"], exp_name),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
