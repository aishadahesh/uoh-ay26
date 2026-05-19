"""
main.py
-------
Full end-to-end pipeline for Assignment 01 — Signal Reconstruction.

Steps
-----
1. Plot raw clean/noisy signals + a 10-sample window example.
2. Build dataset (10 000 examples, all noise levels).
3. Train FC, RNN, LSTM with MSE loss on the same dataset.
4. Evaluate: overall MSE/MAE/corr, per-frequency MSE, per-noise MSE.
5. Run noise sweep (single-sigma datasets) for noise_vs_mse plot.
6. Generate all plots.
7. Save results/metrics.csv.

Run
---
    cd c:\\Users\\Aisha\\Desktop\\AI\\uoh-ay26
    python src/main.py
    python src/main.py --model fc       # train only FC
    python src/main.py --skip-sweep     # skip noise sweep (faster)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch

from data_generator import get_dataloaders, NOISE_LEVELS
from models         import FCNet, RNNNet, LSTMNet
from train          import train_model, RESULTS_DIR
from evaluate       import (evaluate_model, mse_per_frequency,
                            mse_per_noise_level, noise_sweep, save_metrics_csv)
from plots          import (plot_signals, plot_window_example, plot_training_loss,
                            plot_prediction_vs_true, plot_mse_per_frequency,
                            plot_noise_vs_mse, plot_reconstruction_per_freq)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Signal Reconstruction Pipeline")
parser.add_argument("--model",       default="all",
                    choices=["all", "fc", "rnn", "lstm"],
                    help="Which model(s) to train (default: all)")
parser.add_argument("--epochs",      type=int,   default=100)
parser.add_argument("--n-samples",   type=int,   default=10_000)
parser.add_argument("--batch-size",  type=int,   default=64)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--skip-sweep",  action="store_true",
                    help="Skip the noise-sweep experiment")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*60}")
print(f"  Signal Reconstruction - FC vs RNN vs LSTM")
print(f"  Group: uoh-ay26  |  Device: {DEVICE}")
print(f"  Task: MSE(prediction, clean_window)")
print(f"  Input: [noisy_window(100) | C(4) | sigma] -> clean_window[100]")
print(f"{'='*60}\n")


# ── 1. Visualize signals ──────────────────────────────────────────────────────
print("[1/7] Plotting signals ...")
plot_signals(noise_level=0.10)
plot_window_example()


# ── 2. Dataset ────────────────────────────────────────────────────────────────
print("\n[2/7] Building dataset ...")
train_loader, val_loader, test_loader = get_dataloaders(
    n_samples=args.n_samples,
    noise_levels=NOISE_LEVELS,
    batch_size=args.batch_size,
)
print(f"  Train: {len(train_loader.dataset):>6}  "
      f"Val: {len(val_loader.dataset):>6}  "
      f"Test: {len(test_loader.dataset):>6}")
print(f"  FC  input: [batch, 105]    RNN/LSTM input: [batch, 100, 6]")
print(f"  Target:    [batch, 100]    Loss: MSELoss")


# ── 3. Define models ──────────────────────────────────────────────────────────
model_registry = {}
if args.model in ("all", "fc"):
    model_registry["FC"]   = FCNet  (hidden_size=16)
if args.model in ("all", "rnn"):
    model_registry["RNN"]  = RNNNet (hidden_size=64, num_layers=2)
if args.model in ("all", "lstm"):
    model_registry["LSTM"] = LSTMNet(hidden_size=128, num_layers=2)

print("\n  Model parameter counts:")
for name, model in model_registry.items():
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {name:6s}: {n:>8,} parameters")


# ── 4. Train ──────────────────────────────────────────────────────────────────
print(f"\n[3/7] Training ({args.epochs} epochs each, MSE loss) ...")
histories = {}
for name, model in model_registry.items():
    print(f"\n  --- {name} ---")
    histories[name] = train_model(
        model, train_loader, val_loader,
        n_epochs=args.epochs, lr=args.lr,
        device=DEVICE, model_name=name,
    )


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n[4/7] Evaluating on test set ...")
eval_results  = {}
freq_mse_all  = {}
noise_mse_all = {}

for name, model in model_registry.items():
    ckpt = RESULTS_DIR / f"{name}_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    res  = evaluate_model(model, test_loader, DEVICE, model_name=name)
    eval_results[name]  = res
    freq_mse_all[name]  = mse_per_frequency(res)
    noise_mse_all[name] = mse_per_noise_level(res)

    print(f"  {name}  MSE per frequency: {freq_mse_all[name]}")
    print(f"  {name}  MSE per noise:     {noise_mse_all[name]}")


# ── 6. Noise sweep ────────────────────────────────────────────────────────────
sweep_rows = []
if not args.skip_sweep:
    print("\n[5/7] Noise sweep ...")
    model_configs = [
        {"name": "FC",   "class": FCNet,   "kwargs": {"hidden_size": 16}},
        {"name": "RNN",  "class": RNNNet,  "kwargs": {"hidden_size": 64, "num_layers": 2}},
        {"name": "LSTM", "class": LSTMNet, "kwargs": {"hidden_size": 128, "num_layers": 2}},
    ]
    sweep_rows = noise_sweep(
        model_configs=model_configs,
        noise_levels=NOISE_LEVELS,
        get_loaders_fn=get_dataloaders,
        device=DEVICE,
        n_epochs=30,
        n_samples=4_000,
        batch_size=args.batch_size,
    )
else:
    print("\n[5/7] Skipping noise sweep (--skip-sweep).")


# ── 7. Plots ──────────────────────────────────────────────────────────────────
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


# ── 8. Save metrics ───────────────────────────────────────────────────────────
print("\n[7/7] Saving results ...")
main_rows = [
    {"model": k, "experiment": "main",
     "mse": v["mse"], "mae": v["mae"], "corr": v["corr"]}
    for k, v in eval_results.items()
]
noise_rows_csv = [
    {"model": r["model"], "experiment": "noise_sweep",
     "noise_level": r["noise_level"], "mse": r["mse"]}
    for r in sweep_rows
]
save_metrics_csv(main_rows + noise_rows_csv)

print("\nDone.  Results saved to results/")


print("\n" + "="*60)
print("  All done!  Results are in  results/")
print("  Plots:     results/plots/")
print("  Metrics:   results/metrics.csv")
print("="*60)
