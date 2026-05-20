# PRD — Dataset

## Overview
The dataset consists of synthetic mixed sinusoidal signals.
Four frequency components are generated simultaneously and summed (with per-component noise)
to form the network input. The network must separate one chosen component from the mixture.

**Task type:** Regression / component separation (not classification)  
**Loss:** MSELoss

---

## Signal Generation

For each training example, all four components are generated independently:

```
For k in {0, 1, 2, 3}:
    S_k(t) = A_k · sin(2π · f_k · t + φ_k)
    noisy_k(t) = S_k(t) + η_k(t),   η_k ~ N(0, (σ · A_k)²)

Mixed(t) = noisy_0(t) + noisy_1(t) + noisy_2(t) + noisy_3(t)   <- INPUT
Target = S_{c_idx}(t)                                             <- OUTPUT
```

| Parameter | Distribution | Notes |
|-----------|-------------|-------|
| f_k | {1, 2, 5, 7} Hz | Fixed frequencies |
| A_k | Uniform(0.7, 1.3) | Independent per component |
| φ_k | Uniform(0, 2π) | Independent per component |
| σ | {0.00, 0.10, 0.30, 0.50, 1.00} | Shared across all k |
| Fs | 1000 Hz | Sampling rate |
| Duration | 10 s | 10,000 samples per full signal |

---

## Windowing

A random start position `s` is drawn from `[0, 9900]`.
A **100-sample (100 ms) window** is cut from the full mixed signal:

```
input_window = Mixed[s : s + 100]     <- 100 noisy mixed samples
target_window = S_{c_idx}[s : s + 100] <- 100 clean component samples
```

---

## Selector Encoding

| c_idx | Frequency | One-hot C |
|-------|-----------|----------|
| 0 | 1 Hz | [1, 0, 0, 0] |
| 1 | 2 Hz | [0, 1, 0, 0] |
| 2 | 5 Hz | [0, 0, 1, 0] |
| 3 | 7 Hz | [0, 0, 0, 1] |

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
| FC | `x_flat` | `[batch, 105]` | `[mixed_window(100) \| C(4) \| sigma(1)]` |
| RNN/LSTM | `x_seq` | `[batch, 100, 6]` | per step: `[mixed_val, C1,C2,C3,C4, sigma]` |
| Target | `y` | `[batch, 100]` | clean component window |


## Noise Sweep Experiment

Noise levels tested: `[0%, 5%, 10%, 20%, 30%, 50%]`

Each noise level uses a freshly generated dataset to isolate the effect of noise from training data variance.
