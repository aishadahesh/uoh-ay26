"""Helper functions used by main.py."""

import pandas as pd
import torch

from .data_generator import NOISE_LEVELS, get_dataloaders
from .evaluate import evaluate_model, mse_per_frequency, mse_per_noise_level, save_metrics_csv
from .evaluate_sweep import noise_sweep
from .models import FCNet, LSTMNet, RNNNet
from .plots import (
    plot_mse_per_frequency,
    plot_noise_vs_mse,
    plot_prediction_vs_true,
    plot_reconstruction_per_freq,
    plot_training_loss,
)
from .train import RESULTS_DIR, train_model


def build_models(selected: str) -> dict[str, torch.nn.Module]:
    """Build selected model registry."""
    registry = {}
    if selected in ("all", "fc"):
        registry["FC"] = FCNet(hidden_size=64)
    if selected in ("all", "rnn"):
        registry["RNN"] = RNNNet(hidden_size=64, num_layers=1)
    if selected in ("all", "lstm"):
        registry["LSTM"] = LSTMNet(hidden_size=64, num_layers=1)
    return registry


def train_all(models, train_loader, val_loader, args, device) -> dict:
    """Train every selected model."""
    print(f"\n[3/7] Training ({args.epochs} epochs each, MSE loss) ...")
    histories = {}
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        histories[name] = train_model(
            model, train_loader, val_loader,
            n_epochs=args.epochs, lr=args.lr, device=device, model_name=name
        )
    return histories


def evaluate_all(models, test_loader, device) -> tuple[dict, dict, dict]:
    """Load best checkpoints and evaluate selected models."""
    print("\n[4/7] Evaluating on test set ...")
    eval_results, freq_mse_all, noise_mse_all = {}, {}, {}
    for name, model in models.items():
        ckpt = RESULTS_DIR / f"{name}_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        res = evaluate_model(model, test_loader, device, model_name=name)
        eval_results[name] = res
        freq_mse_all[name] = mse_per_frequency(res)
        noise_mse_all[name] = mse_per_noise_level(res)
        print(f"  {name}  MSE per frequency: {freq_mse_all[name]}")
        print(f"  {name}  MSE per noise:     {noise_mse_all[name]}")
    return eval_results, freq_mse_all, noise_mse_all


def run_noise_sweep(args, device) -> list[dict]:
    """Run optional single-sigma noise sweep."""
    if args.skip_sweep:
        print("\n[5/7] Skipping noise sweep (--skip-sweep).")
        return []
    print("\n[5/7] Noise sweep ...")
    configs = [
        {"name": "FC",   "class": FCNet,   "kwargs": {"hidden_size": 64}},
        {"name": "RNN",  "class": RNNNet,  "kwargs": {"hidden_size": 64, "num_layers": 1}},
        {"name": "LSTM", "class": LSTMNet, "kwargs": {"hidden_size": 64, "num_layers": 1}},
    ]
    return noise_sweep(
        configs, NOISE_LEVELS, get_dataloaders, device,
        n_epochs=30, n_samples=4_000, batch_size=args.batch_size
    )


def make_plots(histories, eval_results, freq_mse_all, sweep_rows) -> None:
    """Generate result plots."""
    print("\n[6/7] Generating plots ...")
    if histories:
        plot_training_loss(histories)
    if eval_results:
        plot_prediction_vs_true(eval_results)
        plot_mse_per_frequency(freq_mse_all)
        plot_reconstruction_per_freq(eval_results)
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        plot_noise_vs_mse(sweep_df)
        print(f"\nNoise sweep results:\n{sweep_df.to_string(index=False)}")


def save_results(eval_results, sweep_rows) -> None:
    """Save main and sweep metrics to CSV."""
    print("\n[7/7] Saving results ...")
    main_rows = [
        {"model": k, "experiment": "main", "mse": v["mse"],
         "mae": v["mae"], "corr": v["corr"]}
        for k, v in eval_results.items()
    ]
    noise_rows = [
        {"model": r["model"], "experiment": "noise_sweep",
         "noise_level": r["noise_level"], "mse": r["mse"]}
        for r in sweep_rows
    ]
    save_metrics_csv(main_rows + noise_rows)
