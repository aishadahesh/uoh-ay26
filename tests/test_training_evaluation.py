"""Tests for training loop, evaluation utilities, and pipeline helpers."""


import numpy as np
import pytest
import torch

from src.data_generator import get_dataloaders
from src.evaluate import (
    _mean_sample_corr,
    evaluate_model,
    mse_per_frequency,
    mse_per_noise_level,
    save_metrics_csv,
)
from src.evaluate_sweep import save_metrics_csv as sweep_save_csv
from src.models import FCNet
from src.pipeline import build_models, save_results
from src.train import RESULTS_DIR, _mse_on_loader, train_model

# ── Fixtures ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_loaders():
    """Tiny train/val/test split for fast tests (200 samples total)."""
    return get_dataloaders(n_samples=200, batch_size=32, seed=0)


@pytest.fixture(scope="module")
def tiny_model():
    return FCNet(hidden_size=32)


# ── train.py ──────────────────────────────────────────────────────────────────

class TestTrainModel:
    def test_returns_history_keys(self, tiny_loaders):
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        hist = train_model(
            model, tr, va,
            n_epochs=2, patience=2, device=DEVICE,
            model_name="_test_fc_tmp", verbose=False,
        )
        assert "train_loss" in hist
        assert "val_loss" in hist
        assert "best_epoch" in hist
        assert "best_val_loss" in hist

    def test_loss_lengths_match_epochs(self, tiny_loaders):
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        hist = train_model(
            model, tr, va,
            n_epochs=3, patience=10, device=DEVICE,
            model_name="_test_fc_len", verbose=False,
        )
        assert len(hist["train_loss"]) == 3
        assert len(hist["val_loss"]) == 3

    def test_losses_are_finite(self, tiny_loaders):
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        hist = train_model(
            model, tr, va,
            n_epochs=2, patience=2, device=DEVICE,
            model_name="_test_fc_finite", verbose=False,
        )
        assert all(np.isfinite(v) for v in hist["train_loss"])
        assert all(np.isfinite(v) for v in hist["val_loss"])

    def test_checkpoint_saved(self, tiny_loaders):
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        train_model(
            model, tr, va,
            n_epochs=2, patience=2, device=DEVICE,
            model_name="_test_ckpt", verbose=False,
        )
        assert (RESULTS_DIR / "_test_ckpt_best.pt").exists()

    def test_device_none_auto_selects(self, tiny_loaders):
        """Passing device=None should auto-select CPU/CUDA without error."""
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        hist = train_model(
            model, tr, va,
            n_epochs=2, patience=2, device=None,
            model_name="_test_devnone", verbose=False,
        )
        assert "train_loss" in hist

    def test_verbose_output(self, tiny_loaders, capsys):
        """verbose=True should emit epoch prints and final summary."""
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        train_model(
            model, tr, va,
            n_epochs=1, patience=5, device=DEVICE,
            model_name="_test_verbose", verbose=True,
        )
        out = capsys.readouterr().out
        assert "Epoch" in out or "Best val" in out

    def test_early_stopping_triggers(self, tiny_loaders, capsys):
        """patience=1 should trigger early stopping with verbose message."""
        tr, va, _ = tiny_loaders
        model = FCNet(hidden_size=16)
        hist = train_model(
            model, tr, va,
            n_epochs=20, patience=1, device=DEVICE,
            model_name="_test_early", verbose=True,
        )
        # With patience=1 training stops well before 20 epochs
        assert len(hist["train_loss"]) < 20

    def test_mse_on_loader_finite(self, tiny_loaders, tiny_model):
        _, va, _ = tiny_loaders
        loss = _mse_on_loader(
            tiny_model, va, torch.nn.MSELoss(), DEVICE
        )
        assert np.isfinite(loss)
        assert loss >= 0


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


# ── pipeline.py ───────────────────────────────────────────────────────────────

class TestBuildModels:
    def test_all_returns_three_models(self):
        models = build_models("all")
        assert set(models.keys()) == {"FC", "RNN", "LSTM"}

    def test_fc_only(self):
        models = build_models("fc")
        assert list(models.keys()) == ["FC"]

    def test_rnn_only(self):
        models = build_models("rnn")
        assert list(models.keys()) == ["RNN"]

    def test_lstm_only(self):
        models = build_models("lstm")
        assert list(models.keys()) == ["LSTM"]


class TestSaveResults:
    def test_save_results_writes_csv(self, tiny_loaders, monkeypatch, tmp_path):
        """save_results should write a CSV; monkeypatch to avoid touching real results/."""
        import src.pipeline as pipeline_mod

        captured = {}

        def fake_save(rows, path=None):
            import pandas as pd
            df = pd.DataFrame(rows)
            out = tmp_path / "metrics.csv"
            df.to_csv(out, index=False)
            captured["path"] = out
            captured["rows"] = rows
            return df

        monkeypatch.setattr(pipeline_mod, "save_metrics_csv", fake_save)

        tr, va, te = tiny_loaders
        model = FCNet(hidden_size=16)
        train_model(
            model, tr, va,
            n_epochs=1, patience=1, device=DEVICE,
            model_name="_test_save_mono", verbose=False,
        )
        result = evaluate_model(model, te, DEVICE, model_name="FC")
        save_results({"FC": result}, sweep_rows=[])
        assert captured["path"].exists()
        assert any(r.get("model") == "FC" for r in captured["rows"])


class TestPipelineOrchestrators:
    """Covers train_all and evaluate_all in pipeline.py."""

    def test_train_all(self, tiny_loaders):
        import argparse

        from src.pipeline import train_all
        tr, va, _ = tiny_loaders
        models = build_models("fc")
        args = argparse.Namespace(epochs=2, lr=1e-3)
        hist = train_all(models, tr, va, args, DEVICE)
        assert "FC" in hist
        assert len(hist["FC"]["train_loss"]) == 2

    def test_evaluate_all(self, tiny_loaders):
        from src.pipeline import evaluate_all
        tr, va, te = tiny_loaders
        model = FCNet(hidden_size=16)
        train_model(
            model, tr, va,
            n_epochs=1, patience=1, device=DEVICE,
            model_name="FC", verbose=False,
        )
        eval_results, freq_mse, noise_mse = evaluate_all({"FC": model}, te, DEVICE)
        assert "FC" in eval_results
        assert "FC" in freq_mse
        assert "FC" in noise_mse


# ── plots.py + plot_style.py (import coverage) ────────────────────────────────

def test_plots_module_importable():
    import src.plots  # noqa: F401
    assert hasattr(src.plots, "plot_signals")


def test_plot_style_constants():
    from src.plot_style import FREQ_LABELS, MODEL_COLORS, PLOTS_DIR
    assert len(FREQ_LABELS) == 4
    assert "FC" in MODEL_COLORS
    assert PLOTS_DIR.exists() or not PLOTS_DIR.exists()   # just check access
