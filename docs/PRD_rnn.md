# PRD — Vanilla RNN (BiRNN)

## Role

BiRNN is the **sequential baseline**. It processes the 100-step mixed-signal sequence
in both directions simultaneously, using gating-free `tanh` recurrence to track which
component to extract at each time step.

---

## Architecture

```
x_seq: (batch, 100, 6)
   └─ Bidirectional RNN(input=6, hidden=64, num_layers=1, nonlinearity=tanh)
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

Weights are initialised with **orthogonal initialisation** for the recurrent matrices
and Xavier uniform for input matrices to improve gradient flow.

---

## Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Hidden size | 64 | Balanced capacity; same for all models |
| Num layers | 1 | Deeper stacking not needed at this scale |
| Bidirectional | True | Doubles effective hidden size; FC cannot match this cheaply |
| LayerNorm | after BiRNN output | Prevents activation drift over 100 steps |
| Optimizer | Adam | lr=1e-3, weight\_decay=1e-4 |
| Loss | MSELoss | Reconstruction task |
| Gradient clip | 1.0 | Prevents exploding gradients from long-range BPTT |

---

## Parameter Count

```
Forward RNN:   (6+64)×64 + 64 = 4,544 + 64 = 4,608
Backward RNN:  same            = 4,608
LayerNorm:     128×2           = 256
Linear(128→1): 128 + 1         = 129
Total:         ≈ 9,601 parameters
```

---

## Vanishing Gradient Limitation

The standard RNN update:
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```
multiplies the gradient by `∂h_t/∂h_{t−1}` at each step. For a 100-step window, gradients
for early steps can vanish if the spectral norm of `W_h · diag(tanh’) < 1`.

Bidirectionality partially mitigates this: the backward pass reads the sequence in reverse,
so any step is at most 50 steps from the nearest sequence end. But it does not eliminate
the issue for slow-varying 1 Hz components that need the full 100-step trend.

---

## Expected Behavior

| Frequency | RNN test MSE | Notes |
|-----------|-------------|-------|
| 1 Hz | 0.321 | Near-linear window; RNN struggles with gentle slope |
| 2 Hz | 0.367 | Some oscillation visible; moderate performance |
| 5 Hz | 0.382 | 5 cycles in window; harder interference separation |
| 7 Hz | 0.375 | Clear oscillation but RNN cannot gate interference |

**Overall test MSE: 0.3609** (worst of the three models).  
RNN loses to LSTM because it has no mechanism to selectively suppress the three
non-selected components that appear in the mixed signal.

