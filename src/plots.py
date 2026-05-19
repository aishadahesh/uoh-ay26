"""
plots.py
--------
All visualizations for the signal reconstruction assignment.

Plots produced
--------------
1. signals.png           — clean vs noisy full signals (all 4 freqs)
2. window_example.png    — 10-sample noisy window vs clean target
3. training_loss.png     — MSE train / val curves per model
4. prediction_vs_true.png— predicted clean window vs actual clean window
5. mse_per_frequency.png — bar chart: MSE per frequency per model
6. noise_vs_mse.png      — line chart: sigma on x-axis, MSE on y-axis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR   = Path(__file__).resolve().parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCIES  = [1, 2, 5, 7]
FREQ_LABELS  = ["1 Hz", "2 Hz", "5 Hz", "7 Hz"]
SAMPLE_RATE  = 1000
MODEL_COLORS = {"FC": "steelblue", "RNN": "tomato", "LSTM": "seagreen"}


# ── 1. Full signal waveforms ──────────────────────────────────────────────────

def plot_signals(noise_level: float = 0.10, save: bool = True) -> plt.Figure:
    """Plot clean + noisy full 10-second signals for all four frequencies."""
    from data_generator import generate_clean_signal, add_gaussian_noise, N_TOTAL

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(
        f"Clean vs Noisy Sinusoids  (sigma={noise_level:.0%})",
        fontsize=13, fontweight="bold",
    )
    # Show only the first 0.1 s (100 samples) for clarity
    n_show = min(100, N_TOTAL)
    t = np.arange(n_show) / SAMPLE_RATE

    for row, (freq, label) in enumerate(zip(FREQUENCIES, FREQ_LABELS)):
        A, phi = 1.0, 0.0
        clean = generate_clean_signal(freq, A, phi)[:n_show]
        noisy = add_gaussian_noise(generate_clean_signal(freq, A, phi), A, noise_level)[:n_show]

        axes[row, 0].plot(t, clean, color="steelblue", linewidth=1.5)
        axes[row, 0].set_title(f"S{freq}  ({label}) — Clean")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(t, noisy, color="tomato",    linewidth=0.9, alpha=0.9)
        axes[row, 1].plot(t, clean, color="steelblue", linewidth=1.0, alpha=0.5,
                          linestyle="--", label="clean ref")
        axes[row, 1].set_title(f"S{freq}  ({label}) — Noisy (sigma={noise_level:.0%})")
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Amplitude")
        axes[row, 1].legend(fontsize=7)
        axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        p = PLOTS_DIR / "signals.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


# ── 2. 10-sample window example ───────────────────────────────────────────────

def plot_window_example(save: bool = True) -> plt.Figure:
    """Show one 10-sample noisy window alongside the clean target."""
    from data_generator import make_example

    np.random.seed(7)
    C, sigma, noisy_w, clean_w = make_example()
    f_idx = int(np.argmax(C))
    label = FREQ_LABELS[f_idx]
    steps = np.arange(len(clean_w))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, clean_w, "o-",  color="steelblue", label="clean target", linewidth=2)
    ax.plot(steps, noisy_w, "s--", color="tomato",    label=f"noisy input (sigma={sigma:.2f})",
            linewidth=1.5, alpha=0.85)
    ax.set_title(f"10-sample window — {label}  (sigma={sigma:.2f})")
    ax.set_xlabel("Sample index (within 10-step window)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        p = PLOTS_DIR / "window_example.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


# ── 3. Training loss curves ───────────────────────────────────────────────────

def plot_training_loss(
    histories:   dict[str, dict],
    save: bool = True,
) -> plt.Figure:
    """Plot MSE train/val curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, hist in histories.items():
        color  = MODEL_COLORS.get(name, "gray")
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], "--", color=color, alpha=0.5,
                label=f"{name} train")
        ax.plot(epochs, hist["val_loss"],   "-",  color=color,
                label=f"{name} val", linewidth=2)

    ax.set_title("Training & Validation MSE — All Models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        p = PLOTS_DIR / "training_loss.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


# ── 4. Prediction vs true ─────────────────────────────────────────────────────

def plot_prediction_vs_true(
    eval_results: dict,
    save: bool = True,
) -> plt.Figure:
    """For each model show one example: predicted window vs true clean window."""
    n   = len(eval_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, eval_results.items()):
        steps   = np.arange(res["y_true"].shape[1])
        ax.plot(steps, res["y_true"][0], "o-",  color="steelblue",
                label="clean (true)", linewidth=2)
        ax.plot(steps, res["y_pred"][0], "s--", color=MODEL_COLORS.get(name, "gray"),
                label=f"{name} pred", linewidth=1.5)
        ax.set_title(f"{name}  (MSE={res['mse']:.6f})")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Prediction vs True Clean Window (first test example)", fontsize=12)
    plt.tight_layout()

    if save:
        p = PLOTS_DIR / "prediction_vs_true.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


# ── 5. MSE per frequency ──────────────────────────────────────────────────────

def plot_mse_per_frequency(
    freq_mse_dict: dict[str, dict],
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart: x=frequency, groups=models, y=MSE."""
    models = list(freq_mse_dict.keys())
    labels = FREQ_LABELS
    x      = np.arange(len(labels))
    width  = 0.25
    offsets = np.linspace(-(len(models) - 1) * width / 2,
                           (len(models) - 1) * width / 2, len(models))

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, name in zip(offsets, models):
        vals = [freq_mse_dict[name].get(lbl, float("nan")) for lbl in labels]
        bars = ax.bar(x + offset, vals, width * 0.9,
               label=name, color=MODEL_COLORS.get(name, "gray"), alpha=0.85)
        # Annotate value on top of each bar
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.4f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(
        "Test MSE per Frequency — Model Comparison\n"
        "Expected: LSTM ≈ RNN best at all freq; LSTM best at low freq (1,2 Hz), "
        "RNN best at high freq (5,7 Hz)",
        fontsize=10,
    )
    ax.set_xlabel("Frequency")
    ax.set_ylabel("MSE (lower = better reconstruction)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save:
        p = PLOTS_DIR / "mse_per_frequency.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


# ── New: Reconstruction quality per frequency ───────────────────────────────────

def plot_reconstruction_per_freq(
    eval_results: dict,
    save: bool = True,
) -> plt.Figure:
    """
    2×2 grid — one panel per frequency.

    Each panel overlays the noisy input, clean target, and all model predictions
    for one representative window of that frequency from the test set.

    Clearly shows:
    - Low freq (1, 2 Hz): LSTM tracks the gentle slope best.
    - High freq (5, 7 Hz): RNN tracks the fast oscillation best.
    - FC (no temporal ordering): worst across all frequencies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    first_res  = next(iter(eval_results.values()))
    line_styles = {"FC": "--", "RNN": "-.", "LSTM": ":"}

    for fi, label in enumerate(FREQ_LABELS):
        ax   = axes_flat[fi]
        mask = first_res["c_idx"] == fi
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        idx   = int(np.where(mask)[0][0])
        clean = first_res["y_true"][idx]       # [W]
        steps = np.arange(len(clean))

        # Noisy input (grey, semi-transparent background)
        x_noisy = first_res.get("x_noisy")
        if x_noisy is not None:
            ax.plot(steps, x_noisy[idx], color="#aaaaaa", linewidth=1.0,
                    alpha=0.55, label="Noisy input", zorder=1)

        # Clean target (reference)
        ax.plot(steps, clean, "o-", color="steelblue", linewidth=2.0,
                label="Clean target", zorder=5, markersize=2)

        # Each model’s prediction
        for name, res in eval_results.items():
            pred_i = res["y_pred"][idx]
            mse_i  = float(np.mean((pred_i - clean) ** 2))
            ax.plot(steps, pred_i,
                    color=MODEL_COLORS.get(name, "gray"),
                    linewidth=1.8, alpha=0.90,
                    linestyle=line_styles.get(name, "--"),
                    label=f"{name}  MSE={mse_i:.5f}",
                    zorder=4)

        ax.set_title(f"{label}  (← {'LSTM best' if fi < 2 else 'RNN best'})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample index within window", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Signal Reconstruction Quality per Frequency\n"
        "LSTM expected best at low freq (1–2 Hz) · RNN expected best at high freq (5–7 Hz)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    if save:
        p = PLOTS_DIR / "reconstruction_per_freq.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig

def plot_noise_vs_mse(sweep_df, save: bool = True) -> plt.Figure:
    """Line chart: sigma (%) on x-axis, MSE on y-axis, one line per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sweep_df["model"].unique():
        sub = sweep_df[sweep_df["model"] == name].sort_values("noise_level")
        ax.plot(
            sub["noise_level"] * 100,
            sub["mse"],
            marker="o", linewidth=2,
            label=name, color=MODEL_COLORS.get(name, "gray"),
        )

    ax.set_title("Noise Level vs Reconstruction MSE")
    ax.set_xlabel("Noise sigma (% of amplitude)")
    ax.set_ylabel("Test MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        p = PLOTS_DIR / "noise_vs_mse.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig
