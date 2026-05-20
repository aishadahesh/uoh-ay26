"""
signals.py
----------
Low-level signal generation primitives.

Exports
-------
one_hot               : one-hot numpy vector
generate_clean_signal : full 10-s sinusoid array
add_gaussian_noise    : amplitude-scaled Gaussian noise
make_example          : one complete separation training example
"""

from __future__ import annotations

import numpy as np

from config import (
    CONTEXT_WINDOW,
    FREQUENCIES,
    N_FREQS,
    N_TOTAL,
    NOISE_LEVELS,
    SAMPLE_RATE,
)


def one_hot(idx: int, n: int = N_FREQS) -> np.ndarray:
    """Return a float32 one-hot vector of length *n* with position *idx* set to 1."""
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def generate_clean_signal(freq: float, A: float, phi: float) -> np.ndarray:
    """
    Build a full 10-second clean sinusoid.

    Returns
    -------
    np.ndarray of shape (N_TOTAL,), dtype float32
        ``A * sin(2*pi * f * t + phi)``
    """
    t = np.arange(N_TOTAL, dtype=np.float64) / SAMPLE_RATE
    return (A * np.sin(2.0 * np.pi * freq * t + phi)).astype(np.float32)


def add_gaussian_noise(
    clean: np.ndarray,
    A:     float,
    sigma: float,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise scaled by amplitude.

    ``std = sigma * |A|`` keeps SNR constant regardless of amplitude.
    Returns a noisy copy; the original array is not mutated.
    """
    if sigma == 0.0:
        return clean.copy()
    noise = np.random.normal(0.0, sigma * abs(A), len(clean)).astype(np.float32)
    return clean + noise


def make_example(
    frequencies:  list = FREQUENCIES,
    noise_levels: list = NOISE_LEVELS,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Generate one denoising training example (single frequency).

    Algorithm
    ---------
    1. Draw target component index ``c_idx`` and one-hot ``C``.
    2. Sample ``sigma`` from *noise_levels*.
    3. Draw ``A ~ U(0.7, 1.3)`` and ``phi ~ U(0, 2*pi)``.
    4. Build clean sinusoid for the selected frequency; add noise.
    5. Cut a random ``CONTEXT_WINDOW``-length window.

    Returns
    -------
    C            : float32 array [N_FREQS], one-hot frequency selector
    sigma        : float, noise level used
    noisy_window : float32 array [CONTEXT_WINDOW], noisy single-frequency window
    clean_window : float32 array [CONTEXT_WINDOW], clean single-frequency window
    """
    c_idx = int(np.random.randint(0, len(frequencies)))
    C     = one_hot(c_idx)
    sigma = float(np.random.choice(noise_levels))
    start = int(np.random.randint(0, N_TOTAL - CONTEXT_WINDOW))

    A   = float(np.random.uniform(0.7, 1.3))
    phi = float(np.random.uniform(0.0, 2.0 * np.pi))

    clean = generate_clean_signal(frequencies[c_idx], A, phi)
    noisy = add_gaussian_noise(clean, A, sigma)

    return C, sigma, noisy[start:start + CONTEXT_WINDOW], clean[start:start + CONTEXT_WINDOW]
