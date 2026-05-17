# PRD — Dataset

## Overview
The dataset consists of synthetic sinusoidal signals at 4 frequencies, with configurable noise.

## Signal Generation

```
x_clean(t) = A · sin(2π · f · t + φ)
x_noisy(t) = x_clean(t) + η(t)
```

- **A**: amplitude sampled from Uniform[1 - ε, 1 + ε], ε = 0.05
- **φ**: phase sampled from Uniform[0, 2π]
- **η(t)**: noise sampled from N(0, (noise_level · A)²) or Uniform[−noise_level·A, +noise_level·A]
- **Fs**: 1000 Hz (sampling rate)
- **Duration**: 10 s → 10 000 samples per full signal

## Windowing

Each dataset sample is a **sliding window** of `SEQ_LEN = 200` samples extracted at a random start position from a freshly generated signal. This creates diverse training examples with varying phase offsets.

## Labels — One-Hot Encoding

| Index | Frequency | One-Hot    |
|-------|-----------|------------|
| 0     | 1 Hz (S1) | [1, 0, 0, 0] |
| 1     | 2 Hz (S2) | [0, 1, 0, 0] |
| 2     | 5 Hz (S5) | [0, 0, 1, 0] |
| 3     | 7 Hz (S7) | [0, 0, 0, 1] |

> Integer labels are used internally for `CrossEntropyLoss`. One-hot representation is used for display and explanation.

## Dataset Splits

| Split | Fraction | Default size (n_per_class=800) |
|-------|----------|--------------------------------|
| Train | 70%      | 2240 samples                   |
| Val   | 15%      | 480 samples                    |
| Test  | 15%      | 480 samples                    |

## Noise Sweep Experiment

Noise levels tested: `[0%, 5%, 10%, 20%, 30%, 50%]`

Each noise level uses a freshly generated dataset to isolate the effect of noise from training data variance.
