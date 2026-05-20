"""Evaluation utilities for reconstruction models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data_generator import CONTEXT_WINDOW, N_FREQS
from .evaluate_sweep import save_metrics_csv as _save_metrics_csv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FREQ_LABELS = ["1 Hz", "2 Hz", "5 Hz", "7 Hz"]
FREQUENCIES = [1, 2, 5, 7]


def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect clean windows, predictions, metadata, and input."""
    model.eval()
    all_true, all_pred, all_cidx, all_sigma, all_noisy = [], [], [], [], []
    with torch.no_grad():
        for x_flat, x_seq, y, sigma_batch in loader:
            x_flat = x_flat.to(device)
            pred = model(x_flat, x_seq.to(device)).cpu().numpy()
            x_np = x_flat.cpu().numpy()
            c_onehot = x_np[:, CONTEXT_WINDOW : CONTEXT_WINDOW + N_FREQS]
            all_true.append(y.numpy())
            all_pred.append(pred)
            all_cidx.append(np.argmax(c_onehot, axis=1))
            all_sigma.append(sigma_batch.squeeze(-1).numpy())
            all_noisy.append(x_np[:, :CONTEXT_WINDOW])
    return (
        np.concatenate(all_true, axis=0),
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_cidx, axis=0),
        np.concatenate(all_sigma, axis=0),
        np.concatenate(all_noisy, axis=0),
    )


def _mean_sample_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Pearson correlation, skipping flat windows to avoid NaN."""
    vals = [
        np.corrcoef(y_true[i], y_pred[i])[0, 1]
        for i in range(len(y_true))
        if np.std(y_true[i]) > 0 and np.std(y_pred[i]) > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str = "model",
) -> dict:
    """Evaluate on test set, print summary, and return metric arrays."""
    y_true, y_pred, c_idx, sigmas, x_noisy = get_predictions(
        model, test_loader, device
    )
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = _mean_sample_corr(y_true, y_pred)
    print(f"\n{'=' * 52}")
    print(f"  {model_name}  |  MSE={mse:.6f}  MAE={mae:.6f}  Corr={corr:.4f}")
    print(f"{'=' * 52}")
    return {
        "model": model_name, "mse": mse, "mae": mae, "corr": corr,
        "y_true": y_true, "y_pred": y_pred, "c_idx": c_idx,
        "sigmas": sigmas, "x_noisy": x_noisy,
    }


def mse_per_frequency(result: dict) -> dict[str, float]:
    """Return {freq_label: mse} for each frequency."""
    y_true, y_pred, c_idx = result["y_true"], result["y_pred"], result["c_idx"]
    out = {}
    for i, label in enumerate(FREQ_LABELS):
        mask = c_idx == i
        out[label] = (
            float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
            if mask.sum() > 0 else float("nan")
        )
    return out


def mse_per_noise_level(result: dict) -> dict[float, float]:
    """Return {sigma: mse} for each unique sigma in the test set."""
    y_true, y_pred, sigmas = result["y_true"], result["y_pred"], result["sigmas"]
    out = {}
    for sigma in np.unique(sigmas):
        mask = sigmas == sigma
        out[float(sigma)] = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    return out


def save_metrics_csv(rows: list[dict], path: Path | None = None) -> pd.DataFrame:
    """Write metrics to results/metrics.csv."""
    return _save_metrics_csv(rows, path or RESULTS_DIR / "metrics.csv")
