"""
tests/test_dataset.py
---------------------
Unit tests for data_generator.py (signal SEPARATION task).

Coverage
--------
- one_hot encoding shape, sum, position
- generate_clean_signal shape, dtype, period
- add_gaussian_noise: sigma=0 returns clean copy; sigma>0 adds noise
- make_example: return types, shapes, window length
  (makes a MIXED window from all 4 components; target is clean component c)
- SignalReconstructionDataset: length, item shapes (x_flat, x_seq, y)
- get_dataloaders: split sizes, batch shapes
- Reproducibility with fixed seed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import torch

from data_generator import (
    CONTEXT_WINDOW,
    FC_INPUT_SIZE,
    FREQUENCIES,
    N_FREQS,
    N_TOTAL,
    NOISE_LEVELS,
    SEQ_FEATURES,
    SignalReconstructionDataset,
    add_gaussian_noise,
    generate_clean_signal,
    get_dataloaders,
    make_example,
    one_hot,
)


# ── one_hot ───────────────────────────────────────────────────────────────────

def test_one_hot_shape():
    for i in range(N_FREQS):
        v = one_hot(i)
        assert v.shape == (N_FREQS,)


def test_one_hot_sum_is_one():
    for i in range(N_FREQS):
        assert one_hot(i).sum() == 1.0


def test_one_hot_correct_position():
    for i in range(N_FREQS):
        v = one_hot(i)
        assert v[i] == 1.0
        assert all(v[j] == 0.0 for j in range(N_FREQS) if j != i)


def test_one_hot_dtype_float32():
    assert one_hot(0).dtype == np.float32


# ── generate_clean_signal ─────────────────────────────────────────────────────

def test_clean_signal_length():
    for freq in FREQUENCIES:
        sig = generate_clean_signal(freq, A=1.0, phi=0.0)
        assert len(sig) == N_TOTAL, f"Expected {N_TOTAL}, got {len(sig)} for {freq} Hz"


def test_clean_signal_dtype():
    sig = generate_clean_signal(1, A=1.0, phi=0.0)
    assert sig.dtype == np.float32


def test_clean_signal_amplitude_bounded():
    for A in [0.7, 1.0, 1.3]:
        sig = generate_clean_signal(5, A=A, phi=0.0)
        assert np.max(np.abs(sig)) <= A + 1e-5


def test_clean_signal_zero_crossings_1hz():
    """At 1 Hz with phi=0, the signal crosses zero at multiples of 500 samples."""
    sig = generate_clean_signal(1, A=1.0, phi=0.0)
    # At sample 500 (t=0.5 s), sin(2π·1·0.5)=sin(π)≈0
    assert abs(sig[500]) < 1e-3


# ── add_gaussian_noise ────────────────────────────────────────────────────────

def test_sigma_zero_returns_clean():
    clean = generate_clean_signal(5, A=1.0, phi=0.0)
    noisy = add_gaussian_noise(clean, A=1.0, sigma=0.0)
    np.testing.assert_array_equal(clean, noisy)


def test_sigma_nonzero_adds_noise():
    np.random.seed(0)
    clean = generate_clean_signal(5, A=1.0, phi=0.0)
    noisy = add_gaussian_noise(clean, A=1.0, sigma=0.1)
    assert not np.array_equal(clean, noisy)


def test_noise_std_roughly_sigma_times_A():
    """Empirical std of noise should be close to sigma·A."""
    np.random.seed(99)
    clean = generate_clean_signal(2, A=1.0, phi=0.0)
    sigma, A = 0.10, 1.0
    noisy = add_gaussian_noise(clean, A=A, sigma=sigma)
    noise = noisy - clean
    empirical_std = float(np.std(noise))
    assert abs(empirical_std - sigma * A) < 0.02


# ── make_example ─────────────────────────────────────────────────────────────

def test_make_example_return_types():
    C, sigma, noisy_w, clean_w = make_example()
    assert isinstance(C,       np.ndarray)
    assert isinstance(sigma,   float)
    assert isinstance(noisy_w, np.ndarray)
    assert isinstance(clean_w, np.ndarray)


def test_make_example_C_shape():
    C, *_ = make_example()
    assert C.shape == (N_FREQS,)


def test_make_example_C_is_one_hot():
    for _ in range(20):
        C, *_ = make_example()
        assert C.sum() == 1.0
        assert set(C.tolist()).issubset({0.0, 1.0})


def test_make_example_window_length():
    _, _, noisy_w, clean_w = make_example()
    assert len(noisy_w) == CONTEXT_WINDOW
    assert len(clean_w) == CONTEXT_WINDOW


def test_make_example_sigma_in_noise_levels():
    for _ in range(30):
        _, sigma, _, _ = make_example()
        assert sigma in NOISE_LEVELS


def test_make_example_mixed_differs_from_clean_component():
    """Mixed window always differs from target clean component (other components are present)."""
    np.random.seed(5)
    for _ in range(10):
        _, sigma, mixed_w, clean_w = make_example(noise_levels=[0.5])
        assert not np.array_equal(mixed_w, clean_w)


# ── SignalReconstructionDataset ───────────────────────────────────────────────

def test_dataset_length():
    n = 200
    ds = SignalReconstructionDataset(n_samples=n, seed=0)
    assert len(ds) == n


def test_dataset_xflat_shape():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    x_flat, x_seq, y, sigma_t = ds[0]
    assert x_flat.shape == (FC_INPUT_SIZE,), f"Expected ({FC_INPUT_SIZE},), got {x_flat.shape}"


def test_dataset_xseq_shape():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    x_flat, x_seq, y, sigma_t = ds[0]
    assert x_seq.shape == (CONTEXT_WINDOW, SEQ_FEATURES), \
        f"Expected ({CONTEXT_WINDOW},{SEQ_FEATURES}), got {x_seq.shape}"


def test_dataset_y_shape():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    _, _, y, _ = ds[0]
    assert y.shape == (CONTEXT_WINDOW,)


def test_dataset_dtypes_float32():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    x_flat, x_seq, y, sigma_t = ds[0]
    assert x_flat.dtype  == torch.float32
    assert x_seq.dtype   == torch.float32
    assert y.dtype       == torch.float32
    assert sigma_t.dtype == torch.float32


def test_dataset_onehot_in_xflat():
    """C is embedded in positions [CONTEXT_WINDOW:CONTEXT_WINDOW+N_FREQS] of x_flat."""
    ds = SignalReconstructionDataset(n_samples=50, seed=1)
    for i in range(50):
        x_flat, _, _, _ = ds[i]
        c_part = x_flat[CONTEXT_WINDOW : CONTEXT_WINDOW + N_FREQS]
        assert abs(c_part.sum().item() - 1.0) < 1e-5


def test_dataset_sigma_t_shape_and_range():
    """sigma_t is a [1] float32 tensor; value must be a valid noise level."""
    ds = SignalReconstructionDataset(n_samples=20, seed=2)
    for i in range(20):
        _, _, _, sigma_t = ds[i]
        assert sigma_t.shape == (1,)
        assert sigma_t.dtype == torch.float32
        assert any(abs(sigma_t.item() - s) < 1e-5 for s in NOISE_LEVELS)


def test_dataset_reproducibility():
    """Same seed → same first example."""
    ds1 = SignalReconstructionDataset(n_samples=5, seed=42)
    ds2 = SignalReconstructionDataset(n_samples=5, seed=42)
    x1, _, y1, _ = ds1[0]
    x2, _, y2, _ = ds2[0]
    assert torch.allclose(x1, x2)
    assert torch.allclose(y1, y2)


def test_dataset_different_seeds_differ():
    ds1 = SignalReconstructionDataset(n_samples=5, seed=0)
    ds2 = SignalReconstructionDataset(n_samples=5, seed=99)
    x1, _, _, _ = ds1[0]
    x2, _, _, _ = ds2[0]
    assert not torch.allclose(x1, x2)


# ── get_dataloaders ───────────────────────────────────────────────────────────

def test_dataloaders_sum_to_total():
    n = 500
    tr, va, te = get_dataloaders(n_samples=n, batch_size=32)
    assert len(tr.dataset) + len(va.dataset) + len(te.dataset) == n


def test_dataloader_batch_shapes():
    tr, _, _ = get_dataloaders(n_samples=100, batch_size=16)
    x_flat, x_seq, y, sigma_t = next(iter(tr))
    assert x_flat.shape[1] == FC_INPUT_SIZE
    assert x_seq.shape[1:] == (CONTEXT_WINDOW, SEQ_FEATURES)
    assert y.shape[1]      == CONTEXT_WINDOW
    assert sigma_t.shape   == (min(16, 100), 1)

