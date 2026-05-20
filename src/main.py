"""End-to-end pipeline for Assignment 01 signal reconstruction."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from data_generator import NOISE_LEVELS, get_dataloaders
from pipeline import (
    build_models,
    evaluate_all,
    make_plots,
    run_noise_sweep,
    save_results,
    train_all,
)
from plots import plot_signals, plot_window_example


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(description="Signal Reconstruction Pipeline")
    parser.add_argument("--model", default="all", choices=["all", "fc", "rnn", "lstm"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip-sweep", action="store_true")
    return parser.parse_args()


def print_header(device: torch.device) -> None:
    """Print run summary."""
    print(f"\n{'=' * 60}")
    print("  Signal Reconstruction - FC vs RNN vs LSTM")
    print(f"  Group: uoh-ay26  |  Device: {device}")
    print("  Task: MSE(prediction, clean_window)")
    print("  Input: [noisy_window(100) | C(4) | sigma] -> clean_window[100]")
    print(f"{'=' * 60}\n")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(device)

    print("[1/7] Plotting signals ...")
    plot_signals(noise_level=0.10)
    plot_window_example()

    print("\n[2/7] Building dataset ...")
    train_loader, val_loader, test_loader = get_dataloaders(
        n_samples=args.n_samples,
        noise_levels=NOISE_LEVELS,
        batch_size=args.batch_size,
    )
    print(f"  Train: {len(train_loader.dataset):>6}  "
          f"Val: {len(val_loader.dataset):>6}  "
          f"Test: {len(test_loader.dataset):>6}")

    models = build_models(args.model)
    print("\n  Model parameter counts:")
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    {name:6s}: {n_params:>8,} parameters")

    histories = train_all(models, train_loader, val_loader, args, device)
    eval_results, freq_mse_all, _ = evaluate_all(models, test_loader, device)
    sweep_rows = run_noise_sweep(args, device)
    make_plots(histories, eval_results, freq_mse_all, sweep_rows)
    save_results(eval_results, sweep_rows)
    print("\nAll done. Results are in results/")


if __name__ == "__main__":
    main()
