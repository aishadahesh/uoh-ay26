"""Low-level signal primitives for mixed-signal component separation.

Exports all shared constants and the four pure-numpy helper functions
used by both the dataset and the visualisation layer.
"""

from __future__ import annotations

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

FREQUENCIES: list[int] = [1, 2, 5, 7]
N_FREQS: int = len(FREQUENCIES)
SAMPLE_RATE: int = 1_000
DURATION: int = 10
N_TOTAL: int = SAMPLE_RATE * DURATION
CONTEXT_WINDOW: int = 100
FC_INPUT_SIZE: int = CONTEXT_WINDOW + N_FREQS + 1   # 105
SEQ_FEATURES: int = 1 + N_FREQS + 1                # 6
NOISE_LEVELS: list[float] = [0.0, 0.1, 0.3, 0.5, 1.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def one_hot(idx: int, n: int = N_FREQS) -> np.ndarray:
    """Return a float32 one-hot vector of length *n*."""
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def generate_clean_signal(freq: float, A: float, phi: float) -> np.ndarray:
    """Full 10-second clean sinusoid: A * sin(2*pi*f*t + phi)."""
    t = np.arange(N_TOTAL, dtype=np.float64) / SAMPLE_RATE
    return (A * np.sin(2.0 * np.pi * freq * t + phi)).astype(np.float32)


def add_gaussian_noise(clean: np.ndarray, A: float, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise with std = sigma * |A|."""
    if sigma == 0.0:
        return clean.copy()
    noise = np.random.normal(0.0, sigma * abs(A), len(clean)).astype(np.float32)
    return clean + noise


def make_example(
    frequencies: list = FREQUENCIES,
    noise_levels: list = NOISE_LEVELS,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """One mixed-signal separation example.

    All four components are generated with independent A and phi, each
    noise-corrupted individually, then summed.  The network receives the
    noisy mixture and must recover the clean selected component.

    Returns
    -------
    C            : float32 [N_FREQS]       one-hot frequency selector
    sigma        : float                   noise level used
    noisy_window : float32 [CONTEXT_WINDOW] noisy mixed-signal slice
    clean_window : float32 [CONTEXT_WINDOW] clean target component slice
    """
    c_idx = int(np.random.randint(0, len(frequencies)))
    C = one_hot(c_idx)
    sigma = float(np.random.choice(noise_levels))
    start = int(np.random.randint(0, N_TOTAL - CONTEXT_WINDOW))

    mixed_noisy = np.zeros(N_TOTAL, dtype=np.float32)
    clean_target: np.ndarray | None = None

    for i, freq in enumerate(frequencies):
        A_i = float(np.random.uniform(0.7, 1.3))
        phi_i = float(np.random.uniform(0.0, 2.0 * np.pi))
        clean_i = generate_clean_signal(freq, A_i, phi_i)
        noisy_i = add_gaussian_noise(clean_i, A_i, sigma)
        mixed_noisy += noisy_i
        if i == c_idx:
            clean_target = clean_i

    return (
        C,
        sigma,
        mixed_noisy[start : start + CONTEXT_WINDOW],
        clean_target[start : start + CONTEXT_WINDOW],
    )
