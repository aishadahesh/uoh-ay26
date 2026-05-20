# PRD — LSTM (Long Short-Term Memory)

## Role

LSTM is the **primary sequential model**. It adds explicit gating on top of RNN to selectively
suppress the three non-selected components that appear in the mixed-signal input. The forget
gate can zero out accumulated influence from unwanted frequencies; the input gate can
amplify the target component’s contribution.

---

## Architecture

```
x_seq: (batch, 100, 6)
   └─ Bidirectional LSTM(input=6, hidden=64, num_layers=1)
         output: (batch, 100, 128)      ← 128 = 64 forward + 64 backward
         └─ LayerNorm(128)              ← stabilises activations across time
               └─ Linear(128 → 1)       ← applied at EVERY time step

Output: y_hat (batch, 100)   ← predicted clean component window
```

The 6-dimensional per-step input is:
```
x_seq[t] = [ mixed_val_t, C1, C2, C3, C4, σ ]
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
| Hidden size | 64 | Balanced capacity; same for all models |
| Num layers | 1 | 4× gate overhead already provides sufficient capacity |
| Bidirectional | True | Forward + backward context; output dim = 128 |
| LayerNorm | after BiLSTM output | Stabilises 128-dim concatenated state |
| Optimizer | Adam | lr=1e-3, weight\_decay=1e-4 |
| Loss | MSELoss | Reconstruction task |
| Gradient clip | 1.0 | Stability |

---

## Parameter Count

```
Forward LSTM:  4 × [(6+64)×64 + 64] = 4 × 4,544 + 4 × 64 = 18,432
Backward LSTM: same                                          = 18,432
LayerNorm:     128 × 2                                       = 256
Linear(128→1): 128 + 1                                      = 129
Total:         ≈ 37,249 parameters
```

LSTM has ~4× more parameters than RNN at the same hidden size (four gates vs. one state update).

---

## Expected Behavior

| Frequency | LSTM test MSE | Notes |
|-----------|-------------|-------|
| 1 Hz | **0.277** | Best model for 1 Hz — gated memory tracks gentle slope |
| 2 Hz | 0.347 | Comparable to FC |
| 5 Hz | 0.370 | LSTM’s cell state helps isolate target component |
| 7 Hz | 0.263 | Clear oscillation; gating suppresses other components |

**Overall test MSE: 0.3131** (between FC and RNN).  
LSTM outperforms RNN by **13%** (0.313 vs 0.361), confirming that the gating advantage
is meaningful even at this small scale. LSTM trails FC because FC’s direct flat-vector
access is more efficient for separating stationary sinusoids at 100-sample windows.

