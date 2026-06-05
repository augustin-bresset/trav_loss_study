"""Traversability metrics for binary LiDAR segmentation.

Three metric classes, each designed to be accumulated incrementally over
batches and computed once per epoch:

BinaryMetrics
    Threshold-based: precision, recall, F1, accuracy, IoU, Fβ (β=2),
    collision_fpr (FP rate on obstacles), specificity.

RankingMetrics
    Score-based: PR-AUC, average precision (AP).
    Accumulates raw logits — no threshold needed.

CalibrationMetrics
    Calibration: ECE (Expected Calibration Error), Brier score.
    Important for planners that consume probability maps.

Typical usage in a val loop::

    bm  = BinaryMetrics()
    rm  = RankingMetrics()
    cal = CalibrationMetrics()

    for batch in val_loader:
        logits = model(batch["sparse_input"])
        labels = batch["labels"]
        bm.update(logits, labels)
        rm.update(logits, labels)
        cal.update(logits, labels)

    metrics = {**bm.compute(), **rm.compute(), **cal.compute()}
"""

from __future__ import annotations

import numpy as np
import torch


class BinaryMetrics:
    """Incremental TP/FP/FN/TN accumulator for binary classification.

    Args:
        threshold: Decision threshold applied to sigmoid(logits).
        beta:      β for Fβ score (default 2 → recall-focused, safety-oriented).
    """

    def __init__(self, threshold: float = 0.5, beta: float = 2.0) -> None:
        self.threshold = threshold
        self.beta      = beta
        # int 0 so first += promotes to a GPU tensor matching the input device.
        self._tp = self._fp = self._fn = self._tn = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            preds = torch.sigmoid(logits) > self.threshold
            y     = labels.bool()
            self._tp += ( preds &  y).sum()
            self._fp += ( preds & ~y).sum()
            self._fn += (~preds &  y).sum()
            self._tn += (~preds & ~y).sum()

    def compute(self) -> dict[str, float]:
        eps = 1e-8
        # Single .item() call per counter — one GPU sync at end of epoch
        TP = self._tp.item()
        FP = self._fp.item()
        FN = self._fn.item()
        TN = self._tn.item()
        total = TP + FP + FN + TN

        prec = TP / (TP + FP + eps)
        rec  = TP / (TP + FN + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        acc  = (TP + TN) / (total + eps)
        iou  = TP / (TP + FP + FN + eps)

        b2   = self.beta ** 2
        fbeta = (1 + b2) * prec * rec / (b2 * prec + rec + eps)

        # Safety-oriented metrics
        # collision_fpr: fraction of true obstacles predicted traversable (FP/(FP+TN))
        # free_space_recall: fraction of traversable correctly identified (= recall)
        collision_fpr    = FP / (FP + TN + eps)
        specificity      = TN / (TN + FP + eps)

        return {
            "precision":      prec,
            "recall":         rec,
            "f1":             f1,
            "acc":            acc,
            "iou":            iou,
            f"f{self.beta}":  fbeta,
            "collision_fpr":  collision_fpr,
            "specificity":    specificity,
        }


class RankingMetrics:
    """Accumulates raw logits for ranking-based metrics.

    Computes PR-AUC and average precision (AP) without thresholding.
    Only valid at the end of an epoch (requires all predictions).
    """

    def __init__(self) -> None:
        self._logits:  list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits.append(logits.detach().cpu().numpy().ravel())
        self._targets.append(labels.cpu().numpy().ravel().astype(np.int32))

    def compute(self) -> dict[str, float]:
        if not self._logits:
            return {"pr_auc": 0.0, "ap": 0.0}

        scores  = np.concatenate(self._logits)
        targets = np.concatenate(self._targets)
        n_pos   = targets.sum()

        if n_pos == 0 or n_pos == len(targets):
            return {"pr_auc": float("nan"), "ap": float("nan")}

        order          = np.argsort(scores)[::-1]
        targets_sorted = targets[order]

        tp_cum = np.cumsum(targets_sorted)
        fp_cum = np.cumsum(1 - targets_sorted)

        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall    = tp_cum / (n_pos + 1e-8)

        # Prepend (recall=0, precision=1) for proper AUC boundary
        precision = np.concatenate([[1.0], precision])
        recall    = np.concatenate([[0.0], recall])

        pr_auc = float(np.abs(np.trapezoid(precision, recall)))
        ap     = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

        return {"pr_auc": pr_auc, "ap": ap}


class CalibrationMetrics:
    """Accumulates probabilities and targets for calibration metrics.

    Computes ECE (Expected Calibration Error) and Brier score.
    Critical for planners that rely on well-calibrated traversability maps.

    Args:
        n_bins: Number of confidence bins for ECE computation.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins   = n_bins
        self._probs:   list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        self._probs.append(probs)
        self._targets.append(labels.cpu().numpy().ravel().astype(np.float32))

    def compute(self) -> dict[str, float]:
        if not self._probs:
            return {"ece": float("nan"), "brier": float("nan")}

        probs   = np.concatenate(self._probs)
        targets = np.concatenate(self._targets)
        n       = len(probs)

        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece  = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if not mask.any():
                continue
            bin_conf = probs[mask].mean()
            bin_acc  = targets[mask].mean()
            ece += mask.sum() / n * abs(bin_conf - bin_acc)

        brier = float(np.mean((probs - targets) ** 2))

        return {"ece": float(ece), "brier": brier}


if __name__ == "__main__":
    import unittest

    class TestBinaryMetrics(unittest.TestCase):
        def test_perfect_predictions(self):
            bm = BinaryMetrics()
            logits  = torch.tensor([ 5.0,  5.0, -5.0, -5.0])
            targets = torch.tensor([1, 1, 0, 0])
            bm.update(logits, targets)
            m = bm.compute()
            self.assertAlmostEqual(m["precision"],     1.0, places=4)
            self.assertAlmostEqual(m["recall"],        1.0, places=4)
            self.assertAlmostEqual(m["iou"],           1.0, places=4)
            self.assertAlmostEqual(m["collision_fpr"], 0.0, places=4)

        def test_all_wrong(self):
            bm = BinaryMetrics()
            logits  = torch.tensor([-5.0, -5.0,  5.0,  5.0])
            targets = torch.tensor([1, 1, 0, 0])
            bm.update(logits, targets)
            m = bm.compute()
            self.assertAlmostEqual(m["recall"], 0.0, places=4)
            self.assertAlmostEqual(m["collision_fpr"], 1.0, places=4)

        def test_fbeta_range(self):
            bm = BinaryMetrics(beta=2.0)
            bm.update(torch.randn(256), torch.randint(0, 2, (256,)))
            m = bm.compute()
            self.assertGreaterEqual(m["f2.0"], 0.0)
            self.assertLessEqual(m["f2.0"],    1.0 + 1e-4)

    class TestRankingMetrics(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(42)
            self.rm = RankingMetrics()
            logits  = torch.randn(512)
            targets = (torch.sigmoid(logits + 0.5) > 0.5).long()
            self.rm.update(logits, targets)

        def test_pr_auc_range(self):
            m = self.rm.compute()
            self.assertGreaterEqual(m["pr_auc"], 0.0)
            self.assertLessEqual(m["pr_auc"],    1.0 + 1e-4)

        def test_perfect_auc(self):
            rm = RankingMetrics()
            logits  = torch.tensor([ 5.0,  5.0, -5.0, -5.0])
            targets = torch.tensor([1, 1, 0, 0])
            rm.update(logits, targets)
            m = rm.compute()
            self.assertAlmostEqual(m["pr_auc"], 1.0, places=3)

    class TestCalibrationMetrics(unittest.TestCase):
        def test_perfect_calibration(self):
            # Perfect model → low ECE and Brier
            cal = CalibrationMetrics()
            logits  = torch.tensor([ 5.0,  5.0, -5.0, -5.0])
            targets = torch.tensor([1, 1, 0, 0])
            cal.update(logits, targets)
            m = cal.compute()
            self.assertLess(m["ece"],   0.1)
            self.assertLess(m["brier"], 0.1)

        def test_brier_range(self):
            cal = CalibrationMetrics()
            cal.update(torch.randn(256), torch.randint(0, 2, (256,)))
            m = cal.compute()
            self.assertGreaterEqual(m["brier"], 0.0)
            self.assertLessEqual(m["brier"],    1.0 + 1e-4)

    unittest.main()
