"""Basic signal, window, training, and prediction plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import FREQ_LABELS, FREQUENCIES, MODEL_COLORS, PLOTS_DIR, SAMPLE_RATE


def plot_signals(noise_level: float = 0.10, save: bool = True) -> plt.Figure:
    """Plot clean + noisy full signals for all four frequencies."""
    from .data_generator import N_TOTAL, add_gaussian_noise, generate_clean_signal

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f"Clean vs Noisy Sinusoids (sigma={noise_level:.0%})", fontsize=13)
    n_show = min(100, N_TOTAL)
    t = np.arange(n_show) / SAMPLE_RATE
    for row, (freq, label) in enumerate(zip(FREQUENCIES, FREQ_LABELS)):
        clean_full = generate_clean_signal(freq, 1.0, 0.0)
        clean = clean_full[:n_show]
        noisy = add_gaussian_noise(clean_full, 1.0, noise_level)[:n_show]
        axes[row, 0].plot(t, clean, color="steelblue", linewidth=1.5)
        axes[row, 0].set_title(f"S{freq} ({label}) - Clean")
        axes[row, 1].plot(t, noisy, color="tomato", linewidth=0.9, alpha=0.9)
        axes[row, 1].plot(t, clean, color="steelblue", linewidth=1.0,
                          alpha=0.5, linestyle="--", label="clean ref")
        axes[row, 1].set_title(f"S{freq} ({label}) - Noisy sigma={noise_level:.0%}")
        for ax in axes[row]:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        axes[row, 1].legend(fontsize=7)
    plt.tight_layout()
    if save:
        p = PLOTS_DIR / "signals.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


def plot_window_example(save: bool = True) -> plt.Figure:
    """Show one noisy window alongside the clean target."""
    from .data_generator import make_example

    np.random.seed(7)
    C, sigma, noisy_w, clean_w = make_example()
    steps = np.arange(len(clean_w))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, clean_w, "o-", color="steelblue", label="clean target", linewidth=2)
    ax.plot(steps, noisy_w, "s--", color="tomato",
            label=f"noisy input (sigma={sigma:.2f})", linewidth=1.5, alpha=0.85)
    ax.set_title(f"Window example - {FREQ_LABELS[int(np.argmax(C))]}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        p = PLOTS_DIR / "window_example.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


def plot_training_loss(histories: dict[str, dict], save: bool = True) -> plt.Figure:
    """Plot MSE train/val curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, hist in histories.items():
        color = MODEL_COLORS.get(name, "gray")
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], "--", color=color, alpha=0.5,
                label=f"{name} train")
        ax.plot(epochs, hist["val_loss"], "-", color=color,
                label=f"{name} val", linewidth=2)
    ax.set_title("Training & Validation MSE - All Models")
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


def plot_prediction_vs_true(eval_results: dict, save: bool = True) -> plt.Figure:
    """For each model show one example: predicted window vs true clean window."""
    fig, axes = plt.subplots(1, len(eval_results), figsize=(6 * len(eval_results), 4))
    axes = [axes] if len(eval_results) == 1 else axes
    for ax, (name, res) in zip(axes, eval_results.items()):
        steps = np.arange(res["y_true"].shape[1])
        ax.plot(steps, res["y_true"][0], "o-", color="steelblue", label="clean", linewidth=2)
        ax.plot(steps, res["y_pred"][0], "s--", color=MODEL_COLORS.get(name, "gray"),
                label=f"{name} pred", linewidth=1.5)
        ax.set_title(f"{name} (MSE={res['mse']:.6f})")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        p = PLOTS_DIR / "prediction_vs_true.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig
