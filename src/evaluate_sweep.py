"""Noise sweep and CSV export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


def noise_sweep(
    model_configs: list[dict],
    noise_levels: list[float],
    get_loaders_fn,
    device: torch.device,
    n_epochs: int = 30,
    n_samples: int = 4_000,
    batch_size: int = 64,
    verbose: bool = False,
) -> list[dict]:
    """Train each model from scratch for each sigma and record test MSE."""
    from evaluate import evaluate_model
    from train import RESULTS_DIR, train_model

    results = []
    for cfg in model_configs:
        name, model_cls, kwargs = cfg["name"], cfg["class"], cfg["kwargs"]
        for sigma in noise_levels:
            print(f"  [{name}] sigma={sigma:.2f} ...", end=" ", flush=True)
            tr, va, te = get_loaders_fn(
                n_samples=n_samples, noise_levels=[sigma], batch_size=batch_size
            )
            model = model_cls(**kwargs)
            train_model(
                model,
                tr,
                va,
                n_epochs=n_epochs,
                device=device,
                model_name=f"{name}_sw{int(sigma * 100)}",
                verbose=verbose,
            )
            ckpt = RESULTS_DIR / f"{name}_sw{int(sigma * 100)}_best.pt"
            if ckpt.exists():
                model.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=True)
                )
            res = evaluate_model(model, te, device, model_name=f"{name}_sw")
            results.append({"model": name, "noise_level": sigma, "mse": res["mse"]})
            print(f"MSE={res['mse']:.6f}")
    return results


def save_metrics_csv(rows: list[dict], path: Path) -> pd.DataFrame:
    """Write metric rows to CSV and return the DataFrame."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nMetrics saved -> {path}")
    return df
