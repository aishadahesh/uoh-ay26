"""Tests for training loop and _mse_on_loader."""

import numpy as np
import torch

from src.data_generator import get_dataloaders
from src.models import FCNet
from src.train import RESULTS_DIR, _mse_on_loader, train_model

DEVICE = torch.device("cpu")


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

    def test_early_stopping_triggers(self, capsys):
        """patience=1 should trigger early stopping with verbose message."""
        # Use a tiny 20-sample dataset so the model overfits and val MSE rises.
        tr, va, _ = get_dataloaders(n_samples=20, batch_size=10, seed=42)
        model = FCNet(hidden_size=16)
        hist = train_model(
            model, tr, va,
            n_epochs=100, patience=1, device=DEVICE,
            model_name="_test_early", verbose=True,
        )
        # With patience=1 on a 20-sample dataset, early stopping triggers before 100 epochs
        assert len(hist["train_loss"]) < 100

    def test_mse_on_loader_finite(self, tiny_loaders, tiny_model):
        _, va, _ = tiny_loaders
        loss = _mse_on_loader(
            tiny_model, va, torch.nn.MSELoss(), DEVICE
        )
        assert np.isfinite(loss)
        assert loss >= 0
