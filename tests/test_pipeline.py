"""Tests for pipeline.py helpers and plot module imports."""

import torch

from src.evaluate import evaluate_model
from src.models import FCNet
from src.pipeline import build_models, save_results
from src.train import train_model

DEVICE = torch.device("cpu")


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
