"""Comparison plots for metrics and reconstruction examples."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_style import FREQ_LABELS, MODEL_COLORS, PLOTS_DIR


def plot_mse_per_frequency(freq_mse_dict: dict[str, dict], save: bool = True):
    """Grouped bar chart: x=frequency, groups=models, y=MSE."""
    models = list(freq_mse_dict.keys())
    x = np.arange(len(FREQ_LABELS))
    width = 0.25
    offsets = np.linspace(-(len(models) - 1) * width / 2,
                          (len(models) - 1) * width / 2, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, name in zip(offsets, models):
        vals = [freq_mse_dict[name].get(lbl, float("nan")) for lbl in FREQ_LABELS]
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=name, color=MODEL_COLORS.get(name, "gray"), alpha=0.85)
        for bar, value in zip(bars, vals):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(FREQ_LABELS)
    ax.set_title("Test MSE per Frequency - Model Comparison")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save:
        p = PLOTS_DIR / "mse_per_frequency.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


def plot_reconstruction_per_freq(eval_results: dict, save: bool = True):
    """One panel per frequency with noisy input, clean target, and predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    first_res = next(iter(eval_results.values()))
    line_styles = {"FC": "--", "RNN": "-.", "LSTM": ":"}
    for freq_idx, label in enumerate(FREQ_LABELS):
        ax = axes.flatten()[freq_idx]
        mask = first_res["c_idx"] == freq_idx
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        idx = int(np.where(mask)[0][0])
        clean = first_res["y_true"][idx]
        steps = np.arange(len(clean))
        x_noisy = first_res.get("x_noisy")
        if x_noisy is not None:
            ax.plot(steps, x_noisy[idx], color="#aaaaaa", linewidth=1.0,
                    alpha=0.55, label="Noisy input", zorder=1)
        ax.plot(steps, clean, "o-", color="steelblue", linewidth=2.0,
                label="Clean target", zorder=5, markersize=2)
        for name, res in eval_results.items():
            pred = res["y_pred"][idx]
            mse = float(np.mean((pred - clean) ** 2))
            ax.plot(steps, pred, color=MODEL_COLORS.get(name, "gray"),
                    linewidth=1.8, alpha=0.90, linestyle=line_styles.get(name, "--"),
                    label=f"{name} MSE={mse:.5f}", zorder=4)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample index within window")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save:
        p = PLOTS_DIR / "reconstruction_per_freq.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved -> {p}")
    return fig


def plot_noise_vs_mse(sweep_df, save: bool = True):
    """Line chart: sigma (%) on x-axis, MSE on y-axis, one line per model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in sweep_df["model"].unique():
        sub = sweep_df[sweep_df["model"] == name].sort_values("noise_level")
        ax.plot(sub["noise_level"] * 100, sub["mse"], marker="o", linewidth=2,
                label=name, color=MODEL_COLORS.get(name, "gray"))
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
