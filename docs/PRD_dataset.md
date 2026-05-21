# PRD — Dataset

## Overview
The dataset consists of synthetic single-component sinusoidal signals with additive
Gaussian noise. The network receives the noisy version of one sinusoid and must
reconstruct the clean version — a **denoising regression** task.

**Task type:** Regression / denoising (not classification, not source separation)
**Loss:** MSELoss

---

## Signal Generation

For each training example, a single sinusoidal component is generated:

```
A    ~ Uniform(0.7, 1.3)
φ    ~ Uniform(0, 2π)
f    ← randomly chosen from {1, 2, 5, 7} Hz

clean(t) = A · sin(2π · f · t + φ)
noisy(t) = clean(t) + η(t),   η ~ N(0, (σ · A)²)

INPUT:  noisy_window  = noisy[s : s + 100]   ← noisy 100-sample slice
TARGET: clean_window  = clean[s : s + 100]   ← clean 100-sample slice
```

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| f | {1, 2, 5, 7} Hz | Randomly chosen; encoded as one-hot C |
| A | Uniform(0.7, 1.3) | Random amplitude jitter |
| φ | Uniform(0, 2π) | Random phase |
| σ | {0.00, 0.10, 0.30, 0.50, 1.00} | Noise level as fraction of A |
| Fs | 1000 Hz | Sampling rate |
| Duration | 10 s | 10,000 samples per full signal |

---

## Windowing

A random start position `s` is drawn from `[0, 9900]`.
A **100-sample (100 ms) window** is cut from the full signal:

```
noisy_window  = noisy[s : s + 100]    ← 100 noisy samples (network INPUT)
clean_window  = clean[s : s + 100]    ← 100 clean samples (TARGET)
```

---

## Selector Encoding

| c_idx | Frequency | One-hot C |
|-------|-----------|----------|
| 0 | 1 Hz | [1, 0, 0, 0] |
| 1 | 2 Hz | [0, 1, 0, 0] |
| 2 | 5 Hz | [0, 0, 1, 0] |
| 3 | 7 Hz | [0, 0, 0, 1] |

The one-hot C and scalar σ are concatenated to the input so the model knows which
frequency to expect and how noisy the signal is.

---

## Dataset Splits

| Split | Fraction | Default size (n=10,000) |
|-------|----------|------------------------|
| Train | 70% | 7,000 samples |
| Val | 15% | 1,500 samples |
| Test | 15% | 1,500 samples |

Seed = 42 for reproducibility.

---

## Input / Output Shapes

| Path | Tensor | Shape | Contents |
|------|--------|-------|----------|
| FC | `x_flat` | `[batch, 105]` | `[noisy_window(100) \| C(4) \| sigma(1)]` |
| RNN/LSTM | `x_seq` | `[batch, 100, 6]` | per step: `[noisy_val, C1,C2,C3,C4, sigma]` |
| Target | `y` | `[batch, 100]` | clean sinusoidal window |

---

## Noise Sweep Experiment

Noise levels tested: `{0.00, 0.10, 0.30, 0.50, 1.00}`

Each noise level trains a fresh model (30 epochs, 4,000 samples) to isolate the
effect of noise level on reconstruction quality. All three models show strictly
increasing MSE with σ, confirming the expected noise-sensitivity relationship.
