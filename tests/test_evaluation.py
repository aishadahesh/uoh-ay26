"""Tests for evaluate.py and evaluate_sweep.py."""

import numpy as np
import pytest
import torch

from src.evaluate import (
    _mean_sample_corr,
    evaluate_model,
    mse_per_frequency,
    mse_per_noise_level,
    save_metrics_csv,
)
from src.evaluate_sweep import save_metrics_csv as sweep_save_csv
from src.models import FCNet
from src.train import train_model

DEVICE = torch.device("cpu")


# ── evaluate.py ───────────────────────────────────────────────────────────────

class TestEvaluateModel:
    @pytest.fixture(scope="class")
    def trained_result(self, tiny_loaders):
        tr, va, te = tiny_loaders
        model = FCNet(hidden_size=16)
        train_model(
            model, tr, va,
            n_epochs=2, patience=2, device=DEVICE,
            model_name="_test_eval", verbose=False,
        )
        return evaluate_model(model, te, DEVICE, model_name="_test_eval")

    def test_result_keys(self, trained_result):
        for key in ("mse", "mae", "corr", "y_true", "y_pred", "c_idx", "sigmas"):
            assert key in trained_result

    def test_mse_positive(self, trained_result):
        assert trained_result["mse"] >= 0

    def test_shapes_match(self, trained_result):
        n = len(trained_result["y_true"])
        assert trained_result["y_pred"].shape[0] == n
        assert trained_result["c_idx"].shape[0] == n
        assert trained_result["sigmas"].shape[0] == n

    def test_mse_per_frequency_keys(self, trained_result):
        freq_mse = mse_per_frequency(trained_result)
        assert set(freq_mse.keys()) == {"1 Hz", "2 Hz", "5 Hz", "7 Hz"}

    def test_mse_per_frequency_positive(self, trained_result):
        freq_mse = mse_per_frequency(trained_result)
        assert all(v >= 0 or np.isnan(v) for v in freq_mse.values())

    def test_mse_per_noise_level_keys(self, trained_result):
        noise_mse = mse_per_noise_level(trained_result)
        assert all(isinstance(k, float) for k in noise_mse.keys())
        assert all(v >= 0 for v in noise_mse.values())


class TestMeanSampleCorr:
    def test_perfect_correlation(self):
        y = np.random.randn(10, 100)
        assert abs(_mean_sample_corr(y, y) - 1.0) < 1e-6

    def test_flat_window_skipped(self):
        y_true = np.ones((5, 100))
        y_pred = np.ones((5, 100))
        # Should not raise; flat windows are skipped → returns 0.0
        result = _mean_sample_corr(y_true, y_pred)
        assert result == 0.0


class TestSaveMetricsCsv:
    def test_creates_file(self, tmp_path):
        rows = [{"model": "FC", "experiment": "main", "mse": 0.3}]
        out = tmp_path / "metrics.csv"
        df = save_metrics_csv(rows, path=out)
        assert out.exists()
        assert len(df) == 1
        assert df["model"].iloc[0] == "FC"


# ── evaluate_sweep.py ─────────────────────────────────────────────────────────

class TestSweepSaveMetricsCsv:
    def test_creates_file(self, tmp_path):
        rows = [
            {"model": "FC", "noise_level": 0.1, "mse": 0.25},
            {"model": "RNN", "noise_level": 0.1, "mse": 0.35},
        ]
        out = tmp_path / "sweep.csv"
        df = sweep_save_csv(rows, path=out)
        assert out.exists()
        assert len(df) == 2
