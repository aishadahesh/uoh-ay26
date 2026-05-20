"""Compatibility exports for all plotting functions."""

from .plot_basic import (
    plot_prediction_vs_true,
    plot_signals,
    plot_training_loss,
    plot_window_example,
)
from .plot_comparison import (
    plot_mse_per_frequency,
    plot_noise_vs_mse,
    plot_reconstruction_per_freq,
)

__all__ = [
    "plot_signals",
    "plot_window_example",
    "plot_training_loss",
    "plot_prediction_vs_true",
    "plot_mse_per_frequency",
    "plot_noise_vs_mse",
    "plot_reconstruction_per_freq",
]
