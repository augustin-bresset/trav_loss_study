from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
    pred = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(bool)
    terr = terrain.cpu().numpy().astype(bool)
    return float((pred == terr).mean())


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        log_dir: str | Path = "runs",
        save_dir: str | Path = "checkpoints",
    ) -> None:
        self.model        = model
        self.criterion    = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        self.log_dir  = str(log_dir)
        self.save_dir = str(save_dir)
        os.makedirs(self.log_dir,  exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def fit(self) -> float:
        best_f1 = 0.0
        for epoch in range(1, self.epochs + 1):
            tm = self._train_epoch(epoch)
            for k, v in tm.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            vm = self._val_epoch(epoch)
            for k, v in vm.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            print(
                f"  val  loss={vm['loss']:.4f}  f1={vm['f1']:.4f}"
                f"  alt_agree={vm['alt_agreement']:.4f}"
            )

            if vm["f1"] > best_f1:
                best_f1 = vm["f1"]
                ckpt = os.path.join(self.save_dir, "best.pth")
                torch.save(self.model.state_dict(), ckpt)
                print(f"  → best saved (f1={best_f1:.4f})")

        self.writer.close()
        print(f"\nDone. Best val F1={best_f1:.4f}")
        return best_f1

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        losses, metrics = [], BinaryMetrics()
        use_amp = self.device.type == "cuda"

        for i, batch in enumerate(self.train_loader):
            st     = batch["sparse_input"].to(self.device)
            labels = batch["labels"].to(self.device)
            if st.feats.shape[0] == 0:
                continue

            self.optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(st)
                    loss   = self.criterion(logits, labels.float())
            except Exception:
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            metrics.update(logits.detach(), labels)
            print(
                f"\rEpoch {epoch}/{self.epochs} [{i+1}/{len(self.train_loader)}]"
                f"  loss={np.mean(losses):.4f}",
                end="",
            )

        print()
        self.scheduler.step()

        result = metrics.compute()
        result["loss"] = float(np.mean(losses)) if losses else float("nan")
        return result

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        losses, metrics = [], BinaryMetrics()
        alt_agreements: list[float] = []
        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            for batch in self.val_loader:
                st     = batch["sparse_input"].to(self.device)
                labels = batch["labels"].to(self.device)
                alt    = batch.get("alt_labels")
                if st.feats.shape[0] == 0:
                    continue
                try:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = self.model(st)
                        loss   = self.criterion(logits, labels.float())
                except Exception:
                    continue

                losses.append(loss.item())
                metrics.update(logits, labels)
                if alt is not None:
                    alt_agreements.append(terrain_agreement(logits.cpu(), alt))

        result = metrics.compute()
        result["loss"]          = float(np.mean(losses)) if losses else float("nan")
        result["alt_agreement"] = float(np.mean(alt_agreements)) if alt_agreements else 0.0
        return result
