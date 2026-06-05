from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .metrics import BinaryMetrics, RankingMetrics, CalibrationMetrics


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
        full_metrics: bool = False,
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

        self.full_metrics = full_metrics
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

            gt_sem = vm.get("gt_sem_agree", float("nan"))
            gt_sem_str = f"  gt↔sem={gt_sem:.3f}" if not np.isnan(gt_sem) else ""
            print(
                f"  val  loss={vm['loss']:.4f}  f1={vm['f1']:.4f}"
                f"  iou={vm['iou']:.4f}  pr_auc={vm['pr_auc']:.4f}"
                f"  ece={vm['ece']:.4f}  coll_fpr={vm['collision_fpr']:.4f}"
                + gt_sem_str
            )

            if vm["f1"] > best_f1:
                best_f1 = vm["f1"]
                ckpt = os.path.join(self.save_dir, "best.pth")
                # Pour SparseTravNetPUNCE : sauvegarde uniquement le backbone
                sd = (self.model.backbone.state_dict()
                      if hasattr(self.model, "backbone")
                      else self.model.state_dict())
                torch.save(sd, ckpt)
                print(f"  → best saved (f1={best_f1:.4f})")

        self.writer.close()
        print(f"\nDone. Best val F1={best_f1:.4f}")
        return best_f1

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        loss_acc  = torch.tensor(0.0, device=self.device)
        n_batches = 0
        metrics   = BinaryMetrics()
        use_amp   = self.device.type == "cuda"

        for i, batch in enumerate(self.train_loader):
            st     = batch["sparse_input"].to(self.device)
            labels = batch["labels"].to(self.device)
            if st.feats.shape[0] == 0:
                continue

            self.optimizer.zero_grad()
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out  = self.model(st)
                    loss = self.criterion(out, labels.float())
            except Exception:
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            logits = out[0].detach() if isinstance(out, tuple) else out.detach()
            loss_acc += loss.detach()
            n_batches += 1
            metrics.update(logits, labels)
            print(
                f"\rEpoch {epoch}/{self.epochs} [{i+1}/{len(self.train_loader)}]",
                end="",
            )

        print()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler.step()

        result = metrics.compute()
        result["loss"] = (loss_acc / n_batches).item() if n_batches else float("nan")
        return result

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        loss_acc       = torch.tensor(0.0, device=self.device)
        n_batches      = 0
        metrics        = BinaryMetrics()
        ranking        = RankingMetrics()
        calibration    = CalibrationMetrics()
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
                        out  = self.model(st)
                        loss = self.criterion(out, labels.float())
                except Exception:
                    continue

                logits = out[0] if isinstance(out, tuple) else out
                loss_acc += loss.detach()
                n_batches += 1

                eval_labels = alt.to(self.device) if alt is not None else labels
                metrics.update(logits, eval_labels)
                if self.full_metrics:
                    ranking.update(logits, eval_labels)
                    calibration.update(logits, eval_labels)
                    if alt is not None:
                        alt_agreements.append(terrain_agreement(logits.cpu(), labels.cpu()))

        result = {
            **metrics.compute(),
            **ranking.compute(),
            **calibration.compute(),
            "loss":           (loss_acc / n_batches).item() if n_batches else float("nan"),
            "gt_sem_agree":   float(np.mean(alt_agreements)) if alt_agreements else float("nan"),
        }
        return result
