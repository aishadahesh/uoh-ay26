"""Core unit tests for signal primitives and example generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from data_generator import (
    CONTEXT_WINDOW,
    FREQUENCIES,
    N_FREQS,
    N_TOTAL,
    NOISE_LEVELS,
    add_gaussian_noise,
    generate_clean_signal,
    make_example,
    one_hot,
)


def test_one_hot_shape():
    for i in range(N_FREQS):
        assert one_hot(i).shape == (N_FREQS,)


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


def test_clean_signal_length():
    for freq in FREQUENCIES:
        assert len(generate_clean_signal(freq, A=1.0, phi=0.0)) == N_TOTAL


def test_clean_signal_dtype():
    assert generate_clean_signal(1, A=1.0, phi=0.0).dtype == np.float32


def test_clean_signal_amplitude_bounded():
    for A in [0.7, 1.0, 1.3]:
        sig = generate_clean_signal(5, A=A, phi=0.0)
        assert np.max(np.abs(sig)) <= A + 1e-5


def test_clean_signal_zero_crossings_1hz():
    """At 1 Hz with phi=0, sample 500 is near a zero crossing."""
    sig = generate_clean_signal(1, A=1.0, phi=0.0)
    assert abs(sig[500]) < 1e-3


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
    """Empirical std of noise should be close to sigma * A."""
    np.random.seed(99)
    clean = generate_clean_signal(2, A=1.0, phi=0.0)
    noisy = add_gaussian_noise(clean, A=1.0, sigma=0.10)
    assert abs(float(np.std(noisy - clean)) - 0.10) < 0.02


def test_make_example_return_types():
    C, sigma, noisy_w, clean_w = make_example()
    assert isinstance(C, np.ndarray)
    assert isinstance(sigma, float)
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


def test_make_example_noisy_differs_when_sigma_positive():
    np.random.seed(5)
    for _ in range(10):
        _, _, noisy_w, clean_w = make_example(noise_levels=[0.5])
        assert not np.array_equal(noisy_w, clean_w)
