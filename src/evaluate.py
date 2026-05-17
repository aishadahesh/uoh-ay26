"""
evaluate.py
-----------
Evaluation utilities for signal RECONSTRUCTION models (MSE-based).

Functions
---------
evaluate_model       — overall MSE, MAE, correlation on test set
mse_per_frequency    — MSE broken down by selected frequency (C)
mse_per_noise_level  — MSE broken down by sigma value
noise_sweep          — full train+eval loop across sigma levels
save_metrics_csv     — write list of metric dicts to results/metrics.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_generator import CONTEXT_WINDOW, N_FREQS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FREQ_LABELS  = ["1 Hz", "2 Hz", "5 Hz", "7 Hz"]
FREQUENCIES  = [1, 2, 5, 7]


# ── Prediction helper ─────────────────────────────────────────────────────────

def get_predictions(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and collect (y_true, y_pred, C_indices, sigmas, x_noisy).

    Returns
    -------
    y_true   : [N, W]  clean windows
    y_pred   : [N, W]  model predictions
    c_idx    : [N]     integer frequency index (0-3)
    sigmas   : [N]     noise level used for each example
    x_noisy  : [N, W]  noisy input windows
    """
    model.eval()
    all_true, all_pred, all_cidx, all_sigma, all_noisy = [], [], [], [], []
    with torch.no_grad():
        for x_flat, x_seq, y, sigma_batch in loader:
            x_flat = x_flat.to(device)
            x_seq  = x_seq.to(device)
            pred   = model(x_flat, x_seq).cpu().numpy()  # [batch, W]
            y_np   = y.numpy()                           # [batch, W]

            # x_flat layout: [noisy(W) | C(N_FREQS)]  — sigma is the 4th DataLoader item
            x_np     = x_flat.cpu().numpy()
            x_noisy  = x_np[:, :CONTEXT_WINDOW]                            # [batch, W]
            c_onehot = x_np[:, CONTEXT_WINDOW : CONTEXT_WINDOW + N_FREQS] # [batch, N_FREQS]
            sigma_v  = sigma_batch.squeeze(-1).numpy()                     # [batch]
            c_idx    = np.argmax(c_onehot, axis=1)                         # [batch]

            all_true.append(y_np)
            all_pred.append(pred)
            all_cidx.append(c_idx)
            all_sigma.append(sigma_v)
            all_noisy.append(x_noisy)

    return (
        np.concatenate(all_true,  axis=0),
        np.concatenate(all_pred,  axis=0),
        np.concatenate(all_cidx,  axis=0),
        np.concatenate(all_sigma, axis=0),
        np.concatenate(all_noisy, axis=0),
    )


# ── Single-model evaluation ───────────────────────────────────────────────────

def evaluate_model(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device:      torch.device,
    model_name:  str = "model",
) -> dict:
    """
    Evaluate on test set, print summary, return metric dict.

    Metrics: overall MSE, MAE, mean Pearson correlation per sample.
    """
    y_true, y_pred, c_idx, sigmas, x_noisy = get_predictions(model, test_loader, device)

    mse   = float(np.mean((y_pred - y_true) ** 2))
    mae   = float(np.mean(np.abs(y_pred - y_true)))
    # per-sample Pearson correlation — skip flat windows (std == 0) to avoid NaN
    corr_vals = [
        np.corrcoef(y_true[i], y_pred[i])[0, 1]
        for i in range(len(y_true))
        if np.std(y_true[i]) > 0 and np.std(y_pred[i]) > 0
    ]
    corr = float(np.mean(corr_vals)) if corr_vals else 0.0

    print(f"\n{'='*52}")
    print(f"  {model_name}  |  MSE={mse:.6f}  MAE={mae:.6f}  Corr={corr:.4f}")
    print(f"{'='*52}")

    return {
        "model":     model_name,
        "mse":       mse,
        "mae":       mae,
        "corr":      corr,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "c_idx":     c_idx,
        "sigmas":    sigmas,
        "x_noisy":   x_noisy,
    }


# ── Breakdown helpers ─────────────────────────────────────────────────────────

def mse_per_frequency(result: dict) -> dict[str, float]:
    """Return {freq_label: mse} for each of the 4 frequencies."""
    y_true, y_pred, c_idx = result["y_true"], result["y_pred"], result["c_idx"]
    out = {}
    for i, label in enumerate(FREQ_LABELS):
        mask = c_idx == i
        if mask.sum() > 0:
            out[label] = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
        else:
            out[label] = float("nan")
    return out


def mse_per_noise_level(result: dict) -> dict[float, float]:
    """Return {sigma: mse} for each unique sigma in the test set."""
    y_true, y_pred, sigmas = result["y_true"], result["y_pred"], result["sigmas"]
    out = {}
    for s in np.unique(sigmas):
        mask = sigmas == s
        out[float(s)] = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
    return out


# ── Noise sweep ───────────────────────────────────────────────────────────────

def noise_sweep(
    model_configs:  list[dict],
    noise_levels:   list[float],
    get_loaders_fn,
    device:         torch.device,
    n_epochs:       int  = 30,
    n_samples:      int  = 4_000,
    batch_size:     int  = 64,
    verbose:        bool = False,
) -> list[dict]:
    """
    For each (model, sigma) pair: train from scratch and record test MSE.

    Parameters
    ----------
    model_configs : list of {"name": str, "class": type, "kwargs": dict}

    Returns
    -------
    List of {"model", "noise_level", "mse"}
    """
    from train import train_model  # local import avoids circularity

    results = []
    for cfg in model_configs:
        name       = cfg["name"]
        ModelClass = cfg["class"]
        kwargs     = cfg["kwargs"]

        for sigma in noise_levels:
            print(f"  [{name}] sigma={sigma:.2f} ...", end=" ", flush=True)
            tr, va, te = get_loaders_fn(
                n_samples=n_samples,
                noise_levels=[sigma],
                batch_size=batch_size,
            )
            m = ModelClass(**kwargs)
            train_model(m, tr, va, n_epochs=n_epochs, device=device,
                        model_name=f"{name}_sw{int(sigma*100)}", verbose=verbose)
            res = evaluate_model(m, te, device, model_name=f"{name}_sw")
            results.append({"model": name, "noise_level": sigma, "mse": res["mse"]})
            print(f"MSE={res['mse']:.6f}")

    return results


# ── CSV export ────────────────────────────────────────────────────────────────

def save_metrics_csv(rows: list[dict], path: Path | None = None) -> pd.DataFrame:
    """Write list of metric dicts to results/metrics.csv."""
    if path is None:
        path = RESULTS_DIR / "metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nMetrics saved → {path}")
    return df

