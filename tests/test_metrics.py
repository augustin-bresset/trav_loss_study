"""Tests for src/metrics.py.

Covers BinaryMetrics, RankingMetrics, CalibrationMetrics.
These are pure-CPU, no GPU needed.
"""

import math

import numpy as np
import pytest
import torch

from src.metrics import BinaryMetrics, CalibrationMetrics, RankingMetrics


# ---------------------------------------------------------------------------
# BinaryMetrics
# ---------------------------------------------------------------------------

class TestBinaryMetrics:
    def test_perfect_predictions(self):
        bm = BinaryMetrics()
        bm.update(torch.tensor([5., 5., -5., -5.]), torch.tensor([1, 1, 0, 0]))
        m = bm.compute()
        assert m["precision"]     == pytest.approx(1.0, abs=1e-4)
        assert m["recall"]        == pytest.approx(1.0, abs=1e-4)
        assert m["iou"]           == pytest.approx(1.0, abs=1e-4)
        assert m["f1"]            == pytest.approx(1.0, abs=1e-4)
        assert m["collision_fpr"] == pytest.approx(0.0, abs=1e-4)

    def test_all_wrong(self):
        bm = BinaryMetrics()
        bm.update(torch.tensor([-5., -5., 5., 5.]), torch.tensor([1, 1, 0, 0]))
        m = bm.compute()
        assert m["recall"]        == pytest.approx(0.0, abs=1e-4)
        assert m["collision_fpr"] == pytest.approx(1.0, abs=1e-4)

    def test_all_predicted_negative(self):
        """Model predicts nothing traversable — recall=0, collision_fpr=0."""
        bm = BinaryMetrics()
        bm.update(torch.full((100,), -5.), torch.randint(0, 2, (100,)))
        m = bm.compute()
        assert m["recall"]        == pytest.approx(0.0, abs=1e-4)
        assert m["collision_fpr"] == pytest.approx(0.0, abs=1e-4)

    def test_fbeta_range(self):
        bm = BinaryMetrics(beta=2.0)
        bm.update(torch.randn(256), torch.randint(0, 2, (256,)))
        m = bm.compute()
        assert 0.0 <= m["f2.0"] <= 1.0 + 1e-4

    def test_accumulates_correctly_across_batches(self):
        """Incremental update must equal single-batch update."""
        torch.manual_seed(0)
        a, ya = torch.randn(100), torch.randint(0, 2, (100,))
        b, yb = torch.randn(100), torch.randint(0, 2, (100,))

        bm_inc = BinaryMetrics()
        bm_inc.update(a, ya)
        bm_inc.update(b, yb)

        bm_full = BinaryMetrics()
        bm_full.update(torch.cat([a, b]), torch.cat([ya, yb]))

        m_inc, m_full = bm_inc.compute(), bm_full.compute()
        assert m_inc["f1"] == pytest.approx(m_full["f1"], abs=1e-6)

    def test_all_values_in_range(self):
        bm = BinaryMetrics()
        bm.update(torch.randn(512), torch.randint(0, 2, (512,)))
        m = bm.compute()
        for key in ("precision", "recall", "f1", "acc", "iou", "collision_fpr", "specificity"):
            assert 0.0 <= m[key] <= 1.0 + 1e-6, f"{key} out of [0, 1]: {m[key]}"


# ---------------------------------------------------------------------------
# RankingMetrics
# ---------------------------------------------------------------------------

class TestRankingMetrics:
    def test_pr_auc_range(self):
        torch.manual_seed(42)
        rm = RankingMetrics()
        logits  = torch.randn(512)
        targets = (torch.sigmoid(logits + 0.5) > 0.5).long()
        rm.update(logits, targets)
        m = rm.compute()
        assert 0.0 <= m["pr_auc"] <= 1.0 + 1e-4

    def test_perfect_ranking(self):
        rm = RankingMetrics()
        rm.update(torch.tensor([5., 5., -5., -5.]), torch.tensor([1, 1, 0, 0]))
        m = rm.compute()
        assert m["pr_auc"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_returns_zero(self):
        rm = RankingMetrics()
        m = rm.compute()
        assert m["pr_auc"] == 0.0

    def test_all_positive_returns_nan(self):
        rm = RankingMetrics()
        rm.update(torch.randn(10), torch.ones(10).long())
        m = rm.compute()
        assert math.isnan(m["pr_auc"])

    def test_all_negative_returns_nan(self):
        rm = RankingMetrics()
        rm.update(torch.randn(10), torch.zeros(10).long())
        m = rm.compute()
        assert math.isnan(m["pr_auc"])

    def test_accumulates_across_batches(self):
        """Splitting into two batches must give the same PR-AUC as one batch."""
        torch.manual_seed(7)
        logits  = torch.randn(200)
        targets = (torch.sigmoid(logits) > 0.5).long()

        rm_one = RankingMetrics()
        rm_one.update(logits, targets)

        rm_two = RankingMetrics()
        rm_two.update(logits[:100], targets[:100])
        rm_two.update(logits[100:], targets[100:])

        assert rm_one.compute()["pr_auc"] == pytest.approx(
            rm_two.compute()["pr_auc"], abs=1e-6
        )


# ---------------------------------------------------------------------------
# CalibrationMetrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        cal = CalibrationMetrics()
        cal.update(torch.tensor([5., 5., -5., -5.]), torch.tensor([1, 1, 0, 0]))
        m = cal.compute()
        assert m["ece"]   < 0.1
        assert m["brier"] < 0.1

    def test_brier_range(self):
        cal = CalibrationMetrics()
        cal.update(torch.randn(256), torch.randint(0, 2, (256,)))
        m = cal.compute()
        assert 0.0 <= m["brier"] <= 1.0 + 1e-4

    def test_empty_returns_nan(self):
        cal = CalibrationMetrics()
        m = cal.compute()
        assert math.isnan(m["ece"])
        assert math.isnan(m["brier"])

    def test_ece_perfect_model(self):
        """High-confidence correct predictions → ECE close to 0."""
        cal = CalibrationMetrics()
        logits  = torch.cat([torch.full((50,), 10.), torch.full((50,), -10.)])
        targets = torch.cat([torch.ones(50), torch.zeros(50)]).long()
        cal.update(logits, targets)
        m = cal.compute()
        assert m["ece"] < 0.05

    def test_ece_worst_model(self):
        """Confidently wrong model → ECE close to 1."""
        cal = CalibrationMetrics()
        logits  = torch.cat([torch.full((50,), -10.), torch.full((50,), 10.)])
        targets = torch.cat([torch.ones(50), torch.zeros(50)]).long()
        cal.update(logits, targets)
        m = cal.compute()
        assert m["ece"] > 0.8
