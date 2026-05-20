# PRD — Product Requirements Document
## Project: Signal Reconstruction using FC, RNN & LSTM
**Group Code:** uoh-ay26  
**Course:** AI Agents Orchestration  
**Assignment:** 01

---

## 1. Problem Statement

Given a **100-sample noisy window** of a sinusoidal signal, together with a **one-hot frequency selector C** and the **noise level sigma**, reconstruct the corresponding **100-sample clean window**.

This is a **regression / denoising** task evaluated with **MSE loss**.

The comparison between FC, RNN, and LSTM demonstrates the difference between:
- **FC (Fully Connected)** — no temporal awareness; treats the 100 values as flat features.
- **RNN (Recurrent Neural Network)** — sequential processing; hidden state h_t propagates context.
- **LSTM (Long Short-Term Memory)** — gated memory; better at preserving signal structure.

---

## 2. Signal Model

```
clean(t) = A · sin(2π · f · t + φ)
noisy(t) = clean(t) + η(t),   η ~ N(0, (sigma · A)²)
```

| Parameter | Value / Range |
|-----------|--------------|
| A         | Uniform(0.7, 1.3) |
| f         | {1, 2, 5, 7} Hz |
| φ (phase) | Uniform(0, 2π) |
| sigma     | {0.00, 0.10, 0.30, 0.50, 1.00} |
| Fs        | 1000 Hz |
| Duration  | 10 s → 10 000 samples per full signal |
| Window    | 100 consecutive samples (100 ms) |

---

## 3. Input / Output Specification

### 3.1 Dataset Item
| Component     | Shape   | Description |
|---------------|---------|-------------|
| C             | (4,)    | One-hot frequency selector |
| sigma         | scalar  | Noise level (fraction of A) |
| noisy_window  | (100,)  | Noisy input window |
| clean_window  | (100,)  | **Target** — clean window |

### 3.2 FC Model
| | Shape |
|---|---|
| Input `x_flat` | (batch, 105) = `[noisy(100) \| C(4) \| sigma(1)]` |
| Output         | (batch, 100) |

### 3.3 RNN / LSTM Model
| | Shape |
|---|---|
| Input `x_seq`  | (batch, 100, 6) — at each step: `[noisy_val, C1, C2, C3, C4, sigma]` |
| Output         | (batch, 100) |

### 3.4 Loss
```
loss = MSE(prediction, clean_window) = mean((ŷ - y)²)
```

---

## 4. One-Hot Encoding

| C vector    | Frequency | Signal |
|-------------|-----------|--------|
| [1, 0, 0, 0] | 1 Hz     | S1     |
| [0, 1, 0, 0] | 2 Hz     | S2     |
| [0, 0, 1, 0] | 5 Hz     | S5     |
| [0, 0, 0, 1] | 7 Hz     | S7     |

---

## 5. Model Architectures

### FC baseline
```
Linear(105, 16) → ReLU → Linear(16, 100)   (~3,400 params)
```

### Bidirectional RNN (2-layer)
```
BiRNN(input=6, hidden=64, layers=2) → LayerNorm(128) → Linear(128, 1) at every step  (~34,400 params)
```

### Bidirectional LSTM (2-layer)
```
BiLSTM(input=6, hidden=128, layers=2) → LayerNorm(256) → Linear(256, 1) at every step  (~535,000 params)
```

---

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| All models MSE at sigma=0.00 | < 1e-3 |
| LSTM MSE at sigma=0.10 | ≤ RNN MSE |
| LSTM MSE at sigma=0.10 | ≤ FC MSE |
| Noise monotone: MSE(0.20) ≥ MSE(0.00) | All models |
| All unit tests pass | `pytest tests/ -v` returns 0 failures |

---

## 7. Non-Functional Requirements

- **Reproducible**: all random seeds fixed (seed=42).
- **Modular**: each source file ≤ ~150 lines.
- **Fast**: full pipeline runs on CPU in < 15 minutes.
- **Tested**: unit tests cover shapes, one-hot, noise, reproducibility.
- **Documented**: README as full lab report with plots and tables.

