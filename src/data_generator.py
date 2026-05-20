"""PyTorch Dataset and DataLoader for the mixed-signal separation task.

All signal-generation logic lives in signals.py.  This module only
handles dataset construction and DataLoader creation.
Everything is re-exported here so existing imports from data_generator
continue to work without modification.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from signals import (  # re-exported for backward compatibility
    CONTEXT_WINDOW,
    FC_INPUT_SIZE,
    FREQUENCIES,
    N_FREQS,
    N_TOTAL,
    NOISE_LEVELS,
    SAMPLE_RATE,
    SEQ_FEATURES,
    add_gaussian_noise,
    generate_clean_signal,
    make_example,
    one_hot,
)

__all__ = [
    "CONTEXT_WINDOW", "FC_INPUT_SIZE", "FREQUENCIES", "N_FREQS", "N_TOTAL",
    "NOISE_LEVELS", "SAMPLE_RATE", "SEQ_FEATURES",
    "add_gaussian_noise", "generate_clean_signal", "make_example", "one_hot",
    "SignalReconstructionDataset", "get_dataloaders",
]


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
        C, sigma = self._C[idx], self._sigma[idx]
        noisy, clean = self._noisy[idx], self._clean[idx]
        sigma_arr = np.array([sigma], dtype=np.float32)
        x_flat = np.concatenate([noisy, C, sigma_arr])
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

