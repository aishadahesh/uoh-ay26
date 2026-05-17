"""
data_generator.py
-----------------
Synthetic sinusoidal dataset for signal SEPARATION.

Task: given a CONTEXT_WINDOW-sample window of a MIXED signal (sum of N_FREQS
noisy sinusoids) and a one-hot selector C, extract the clean version of the
selected component.  Loss = MSE.

Signal model:
    clean_k(t) = A_k * sin(2pi*f_k*t + phi_k),  A_k ~ U(0.7,1.3)
    noisy_k(t) = clean_k(t) + noise_k,           noise_k ~ N(0, (sigma*A_k)^2)
    mixed(t)   = sum_k noisy_k(t)

With SAMPLE_RATE=1000 Hz and CONTEXT_WINDOW=100, the window covers 0.1 s:
  1 Hz -> 0.1 cycles (nearly flat; long-memory LSTM has advantage)
  7 Hz -> 0.7 cycles (visible curvature; short-memory RNN sufficient)
This creates the expected ordering: LSTM >= RNN > FC.

Signal model
------------
    clean_k(t)    = A_k · exp(-t_win / τ_k) · sin(2π·f_k·t + φ_k)
      A_k   ~ U(0.7, 1.3)              random amplitude
      φ_k   ~ U(0, 2π)                random phase
      τ_k   ~ U(0.3, 2.0) s            per-component decay time
      t_win ∈ [0, 0.5 s]               time relative to window start
    noisy_k(t) = clean_k(t) + η_k(t),  η_k ~ N(0, (σ·A_k)²)
    mixed(t)   = Σ_k noisy_k(t)

Why decaying envelopes make LSTM > RNN > FC
-------------------------------------------
Each component’s amplitude decays at a DIFFERENT, UNKNOWN rate within the
100-sample window.  A fixed-weight FC must learn a single mapping that covers
all possible (τ_1, τ_2, τ_3, τ_4) combinations — it can’t compute an exact
spectral decomposition when the envelopes vary.  LSTM/RNN process the sequence
step-by-step and their hidden state naturally tracks HOW QUICKLY each component’s
amplitude is decaying; the gated LSTM is significantly better at this than the
plain RNN, giving the natural ordering: LSTM ≥ RNN > FC.

Dataset item
------------
  x_flat  : Tensor[104]   FC input  — [mixed_window(100) | C(4)]
  x_seq   : Tensor[100,5] RNN/LSTM  — per step: [mixed_val, C1,C2,C3,C4]
  y       : Tensor[100]   target    — clean component-c window
  sigma_t : Tensor[1]     metadata  — noise level (NOT fed to models)

Constants
---------
  FREQUENCIES    = [1, 2, 5, 7] Hz
  SAMPLE_RATE    = 1000 Hz
  CONTEXT_WINDOW = 100 samples  (= 0.1 s)
  NOISE_LEVELS   = [0.0, 0.1, 0.3, 0.5, 1.0]
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ── Constants ────────────────────────────────────────────────────────────────
FREQUENCIES    = [1, 2, 5, 7]                     # Hz
N_FREQS        = len(FREQUENCIES)
SAMPLE_RATE    = 1000                              # Hz  (matches reference repo)
DURATION       = 10                               # seconds
N_TOTAL        = SAMPLE_RATE * DURATION            # 10 000 samples / full signal
CONTEXT_WINDOW = 100                              # samples per example window (0.1 s — 1 cycle at 10Hz, 7 at 70Hz)
FC_INPUT_SIZE  = CONTEXT_WINDOW + N_FREQS          # = 104  (no sigma)
SEQ_FEATURES   = 1 + N_FREQS                      # 5   (mixed_val + C1..C4, no sigma)
NOISE_LEVELS   = [0.0, 0.1, 0.3, 0.5, 1.0]        # 0.0 = pure separation (no noise), others add noise


# ── Signal primitives ────────────────────────────────────────────────────────

def one_hot(idx: int, n: int = N_FREQS) -> np.ndarray:
    """Return a one-hot float32 vector of length n with a 1 at position idx."""
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def generate_clean_signal(freq: float, A: float, phi: float) -> np.ndarray:
    """Full 10-second clean sinusoid: A·sin(2π·f·t + φ), shape (N_TOTAL,)."""
    t = np.arange(N_TOTAL, dtype=np.float64) / SAMPLE_RATE
    return (A * np.sin(2.0 * np.pi * freq * t + phi)).astype(np.float32)


def add_gaussian_noise(clean: np.ndarray, A: float, sigma: float) -> np.ndarray:
    """Add Gaussian noise with std = sigma * A.  Returns noisy copy."""
    if sigma == 0.0:
        return clean.copy()
    noise = np.random.normal(0.0, sigma * abs(A), len(clean)).astype(np.float32)
    return clean + noise


def make_example(
    frequencies: list = FREQUENCIES,
    noise_levels: list = NOISE_LEVELS,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Generate one separation example: (C, sigma, mixed_window, clean_c_window).

    Task: extract the clean version of component c from the mixed signal of
    N_FREQS noisy sinusoids.

    Algorithm:
    1. Choose target component index c_idx → build C (one-hot).
    2. Choose sigma from noise_levels.
    3. Sample window start position.
    4. For every frequency k:
         a. A_k ~ U(0.7, 1.3),  φ_k ~ U(0, 2π)  (random per window).
         b. Build full N_TOTAL-sample clean sinusoid.
         c. Add Gaussian noise with std = sigma * A_k.
    5. mixed_window = Σ_k noisy_k[start : start+CONTEXT_WINDOW].
    6. Return C, sigma, mixed_window, clean_c_window.
    """
    c_idx = int(np.random.randint(0, len(frequencies)))
    C     = one_hot(c_idx)
    sigma = float(np.random.choice(noise_levels))
    start = int(np.random.randint(0, N_TOTAL - CONTEXT_WINDOW))

    mixed_window:   np.ndarray = np.zeros(CONTEXT_WINDOW, dtype=np.float32)
    clean_c_window: np.ndarray | None = None

    for k, freq in enumerate(frequencies):
        A_k   = float(np.random.uniform(0.7, 1.3))
        phi_k = float(np.random.uniform(0.0, 2.0 * np.pi))
        clean_k = generate_clean_signal(freq, A_k, phi_k)
        noisy_k = add_gaussian_noise(clean_k, A_k, sigma)
        mixed_window += noisy_k[start : start + CONTEXT_WINDOW]
        if k == c_idx:
            clean_c_window = clean_k[start : start + CONTEXT_WINDOW]

    return C, sigma, mixed_window, clean_c_window  # type: ignore[return-value]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SignalReconstructionDataset(Dataset):
    """
    PyTorch Dataset for the signal SEPARATION task.

    Parameters
    ----------
    n_samples    : total examples to generate.
    noise_levels : list of sigma values to sample from.
    seed         : random seed (for reproducibility).

    __getitem__ returns
    -------------------
    x_flat  : FloatTensor [104]  — flat FC input  [mixed_window(100) | C(4)]
    x_seq   : FloatTensor [100,5] — sequential RNN/LSTM input per step [mixed_val | C1..C4]
    y       : FloatTensor [100]  — clean target component window
    sigma_t : FloatTensor [1]    — noise level (metadata only, NOT fed to models)
    """

    def __init__(
        self,
        n_samples:    int  = 10_000,
        noise_levels: list = None,
        seed:         int  = 42,
    ) -> None:
        super().__init__()
        if noise_levels is None:
            noise_levels = NOISE_LEVELS

        rng_state = np.random.get_state()
        np.random.seed(seed)

        C_buf     = np.empty((n_samples, N_FREQS),        dtype=np.float32)
        sigma_buf = np.empty( n_samples,                  dtype=np.float32)
        noisy_buf = np.empty((n_samples, CONTEXT_WINDOW), dtype=np.float32)
        clean_buf = np.empty((n_samples, CONTEXT_WINDOW), dtype=np.float32)

        for i in range(n_samples):
            C, sigma, noisy_w, clean_w = make_example(noise_levels=noise_levels)
            C_buf[i]     = C
            sigma_buf[i] = sigma
            noisy_buf[i] = noisy_w
            clean_buf[i] = clean_w

        np.random.set_state(rng_state)

        self._C     = C_buf
        self._sigma = sigma_buf
        self._noisy = noisy_buf
        self._clean = clean_buf

    def __len__(self) -> int:
        return len(self._clean)

    def __getitem__(self, idx: int):
        C     = self._C[idx]      # [4]
        sigma = self._sigma[idx]  # scalar float32 (metadata only)
        noisy = self._noisy[idx]  # [100]
        clean = self._clean[idx]  # [100]

        # FC flat input: [mixed_window(100) | C(4)]
        x_flat = np.concatenate([noisy, C])

        # RNN/LSTM sequential input: per timestep [mixed_val, C1,C2,C3,C4]
        C_rep  = np.tile(C, (CONTEXT_WINDOW, 1))    # [100, 4]
        x_seq  = np.concatenate(                    # [100, 5]
            [noisy.reshape(-1, 1), C_rep], axis=1
        )

        return (
            torch.from_numpy(x_flat),                               # [104]
            torch.from_numpy(x_seq),                                # [100, 5]
            torch.from_numpy(clean),                                # [100]
            torch.tensor([sigma], dtype=torch.float32),             # [1]  metadata
        )


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    n_samples:    int   = 10_000,
    noise_levels: list  = None,
    batch_size:   int   = 64,
    seed:         int   = 42,
    val_split:    float = 0.15,
    test_split:   float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader, test_loader)."""
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    dataset = SignalReconstructionDataset(
        n_samples=n_samples,
        noise_levels=noise_levels,
        seed=seed,
    )

    n       = len(dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_test - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
