# PRD — LSTM (Long Short-Term Memory)

## Role

LSTM is the **highest-capacity sequential model**. It adds explicit gating on top of RNN
to selectively retain and suppress information over the 100-step window. With its large
parameter count (~535K), it is the most expressive model but also the slowest to converge
on CPU at 100 epochs.

---

## Architecture

```
x_seq: (batch, 100, 6)
   └─ Bidirectional LSTM(input=6, hidden=128, num_layers=2)
         output: (batch, 100, 256)      ← 256 = 128 forward + 128 backward
         └─ LayerNorm(256)              ← stabilises activations across time
               └─ Linear(256 → 1)       ← applied at EVERY time step

Output: y_hat (batch, 100)   ← predicted clean window
```

The 6-dimensional per-step input is:
```
x_seq[t] = [ noisy_val_t, C1, C2, C3, C4, σ ]
```
(C and σ are repeated identically at every step.)

Weights are initialised with **orthogonal initialisation** for all recurrent matrices.

---

## LSTM Cell Equations

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   — what to erase from cell state
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   — what new info to write
Cell update:  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell state:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t        — long-term memory
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(C_t)                  — short-term output
```

When `f_t ≈ 1` the cell state is passed unchanged, creating a **direct gradient path**
across 100 steps with no vanishing. This is the fundamental advantage over vanilla RNN.

---

## Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Hidden size | 128 | Larger hidden needed for 4-gate architecture to be expressive |
| Num layers | 2 | Stacked layers add depth for complex temporal patterns |
| Bidirectional | True | Forward + backward context; output dim = 256 |
| LayerNorm | after BiLSTM output | Stabilises 256-dim concatenated state |
| Optimizer | Adam | lr=1e-3, weight\_decay=1e-4 |
| Loss | MSELoss | Denoising regression task |
| Gradient clip | 1.0 | Stability |

---

## Parameter Count

```
Layer 0 forward:   4×[(6+128)×128 + 2×128]   = 4 × 17,664 = 70,656
Layer 0 backward:  4×[(6+128)×128 + 2×128]   = 70,656
Layer 1 forward:   4×[(256+128)×128 + 2×128] = 4 × 49,408 = 197,632
Layer 1 backward:  4×[(256+128)×128 + 2×128] = 197,632
LayerNorm(256):    256×2                       = 512
Linear(256→1):     256 + 1                     = 257
Total:             ≈ 537,345 parameters
```

*(Exact count from PyTorch: 535,297 — minor rounding differences from bias layout.)*

---

## Test Set Results

| Metric | Value |
|--------|-------|
| MSE | 0.007207 |
| MAE | 0.052408 |
| Pearson r | 0.9522 |

**Per-frequency MSE:**

| Frequency | MSE |
|-----------|-----|
| 1 Hz | 0.004841 |
| 2 Hz | 0.006662 |
| 5 Hz | 0.008386 |
| 7 Hz | 0.008945 |

LSTM performs best at 1 Hz (slow-varying target that benefits from long memory) but
struggles at 5 Hz and 7 Hz — the rapid oscillation requires fast adaptation that the
large model has not yet learned in 100 epochs.

---

## Observations

- **Gated cell state:** The forget gate can suppress noisy time steps while the input
  gate amplifies clean signal periods — theoretically the strongest architecture for
  denoising over 100-step sequences.
- **Convergence bottleneck:** ~535K parameters require substantially more than 100 CPU
  epochs to fully converge. With 200+ epochs or GPU training, LSTM would likely
  outperform RNN.
- **Trade-off:** ~16× more parameters than RNN at the cost of slower convergence;
  best deployment scenario is when training time is not a constraint.
