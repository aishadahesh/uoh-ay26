"""Dataset and DataLoader tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from data_generator import (
    CONTEXT_WINDOW,
    FC_INPUT_SIZE,
    N_FREQS,
    NOISE_LEVELS,
    SEQ_FEATURES,
    SignalReconstructionDataset,
    get_dataloaders,
)


def test_dataset_length():
    ds = SignalReconstructionDataset(n_samples=200, seed=0)
    assert len(ds) == 200


def test_dataset_xflat_shape():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    x_flat, _, _, _ = ds[0]
    assert x_flat.shape == (FC_INPUT_SIZE,)


def test_dataset_xseq_shape():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    _, x_seq, _, _ = ds[0]
    assert x_seq.shape == (CONTEXT_WINDOW, SEQ_FEATURES)


def test_dataset_y_shape():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    _, _, y, _ = ds[0]
    assert y.shape == (CONTEXT_WINDOW,)


def test_dataset_dtypes_float32():
    ds = SignalReconstructionDataset(n_samples=10, seed=0)
    x_flat, x_seq, y, sigma_t = ds[0]
    assert x_flat.dtype == torch.float32
    assert x_seq.dtype == torch.float32
    assert y.dtype == torch.float32
    assert sigma_t.dtype == torch.float32


def test_dataset_onehot_in_xflat():
    """C is embedded in positions [CONTEXT_WINDOW:CONTEXT_WINDOW+N_FREQS]."""
    ds = SignalReconstructionDataset(n_samples=50, seed=1)
    for i in range(50):
        x_flat, _, _, _ = ds[i]
        c_part = x_flat[CONTEXT_WINDOW : CONTEXT_WINDOW + N_FREQS]
        assert abs(c_part.sum().item() - 1.0) < 1e-5


def test_dataset_sigma_t_shape_and_range():
    """sigma_t is a [1] float32 tensor and value is a valid noise level."""
    ds = SignalReconstructionDataset(n_samples=20, seed=2)
    for i in range(20):
        _, _, _, sigma_t = ds[i]
        assert sigma_t.shape == (1,)
        assert sigma_t.dtype == torch.float32
        assert any(abs(sigma_t.item() - s) < 1e-5 for s in NOISE_LEVELS)


def test_dataset_reproducibility():
    """Same seed gives the same first example."""
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


def test_dataloaders_sum_to_total():
    tr, va, te = get_dataloaders(n_samples=500, batch_size=32)
    assert len(tr.dataset) + len(va.dataset) + len(te.dataset) == 500


def test_dataloader_batch_shapes():
    tr, _, _ = get_dataloaders(n_samples=100, batch_size=16)
    x_flat, x_seq, y, sigma_t = next(iter(tr))
    assert x_flat.shape[1] == FC_INPUT_SIZE
    assert x_seq.shape[1:] == (CONTEXT_WINDOW, SEQ_FEATURES)
    assert y.shape[1] == CONTEXT_WINDOW
    assert sigma_t.shape == (min(16, 100), 1)
