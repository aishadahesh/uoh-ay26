"""Synthetic sinusoidal dataset for signal separation from a noisy mixture.

Each item returns x_flat [105], x_seq [100, 6], target y [100], and sigma [1].
The INPUT is the noisy *mixed* signal (sum of all 4 components, each with
per-component Gaussian noise).  The TARGET is the clean version of the
selected component.  The one-hot selector C tells the model which component
to extract — identical in spirit to how a prompt steers a language model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

FREQUENCIES = [1, 2, 5, 7]
N_FREQS = len(FREQUENCIES)
SAMPLE_RATE = 1000
DURATION = 10
N_TOTAL = SAMPLE_RATE * DURATION
CONTEXT_WINDOW = 100
FC_INPUT_SIZE = CONTEXT_WINDOW + N_FREQS + 1
SEQ_FEATURES = 1 + N_FREQS + 1
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 1.0]


def one_hot(idx: int, n: int = N_FREQS) -> np.ndarray:
    """Return a one-hot float32 vector of length n."""
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


def generate_clean_signal(freq: float, A: float, phi: float) -> np.ndarray:
    """Full 10-second clean sinusoid: A * sin(2*pi*f*t + phi)."""
    t = np.arange(N_TOTAL, dtype=np.float64) / SAMPLE_RATE
    return (A * np.sin(2.0 * np.pi * freq * t + phi)).astype(np.float32)


def add_gaussian_noise(clean: np.ndarray, A: float, sigma: float) -> np.ndarray:
    """Add Gaussian noise with std = sigma * A."""
    if sigma == 0.0:
        return clean.copy()
    noise = np.random.normal(0.0, sigma * abs(A), len(clean)).astype(np.float32)
    return clean + noise


def make_example(
    frequencies: list = FREQUENCIES,
    noise_levels: list = NOISE_LEVELS,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Generate one training example for component separation.

    All four sinusoidal components are generated with independent random
    amplitudes and phases.  Gaussian noise is added to each component
    individually before they are summed into a single mixed signal.
    The network receives the mixed noisy window and must recover the
    clean version of the component indicated by the one-hot selector C.
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


class SignalReconstructionDataset(Dataset):
    """Pre-generated PyTorch Dataset with reproducible seeded examples."""

    def __init__(
        self,
        n_samples: int = 10_000,
        noise_levels: list | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        noise_levels = NOISE_LEVELS if noise_levels is None else noise_levels
        rng_state = np.random.get_state()
        np.random.seed(seed)

        self._C = np.empty((n_samples, N_FREQS), dtype=np.float32)
        self._sigma = np.empty(n_samples, dtype=np.float32)
        self._noisy = np.empty((n_samples, CONTEXT_WINDOW), dtype=np.float32)
        self._clean = np.empty((n_samples, CONTEXT_WINDOW), dtype=np.float32)

        for i in range(n_samples):
            C, sigma, noisy_w, clean_w = make_example(noise_levels=noise_levels)
            self._C[i] = C
            self._sigma[i] = sigma
            self._noisy[i] = noisy_w
            self._clean[i] = clean_w
        np.random.set_state(rng_state)

    def __len__(self) -> int:
        return len(self._clean)

    def __getitem__(self, idx: int):
        C = self._C[idx]
        sigma = self._sigma[idx]
        noisy = self._noisy[idx]
        clean = self._clean[idx]
        sigma_arr = np.array([sigma], dtype=np.float32)

        # FC input: [window(100) | C(4) | sigma(1)]
        x_flat = np.concatenate([noisy, C, sigma_arr])

        # RNN/LSTM input: per timestep [value, C1, C2, C3, C4, sigma]
        C_rep = np.tile(C, (CONTEXT_WINDOW, 1))
        sigma_rep = np.full((CONTEXT_WINDOW, 1), sigma, dtype=np.float32)
        x_seq = np.concatenate([noisy.reshape(-1, 1), C_rep, sigma_rep], axis=1)

        return (
            torch.from_numpy(x_flat),
            torch.from_numpy(x_seq),
            torch.from_numpy(clean),
            torch.tensor([sigma], dtype=torch.float32),
        )


def get_dataloaders(
    n_samples: int = 10_000,
    noise_levels: list | None = None,
    batch_size: int = 64,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders."""
    dataset = SignalReconstructionDataset(n_samples, noise_levels, seed)
    n_test = int(len(dataset) * test_split)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_test - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )
