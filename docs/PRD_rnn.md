# PRD — Vanilla RNN (BiRNN)

## Role

BiRNN is the **sequential baseline** and **overall winner**. It processes the 100-step
noisy sequence in both directions, using gating-free `tanh` recurrence to denoise
the sinusoidal signal at each time step.

---

## Architecture

```
x_seq: (batch, 100, 6)
   └─ Bidirectional RNN(input=6, hidden=64, num_layers=2, nonlinearity=tanh)
         output: (batch, 100, 128)      ← 128 = 64 forward + 64 backward
         └─ LayerNorm(128)              ← stabilises activations across time
               └─ Linear(128 → 1)       ← applied at EVERY time step

Output: y_hat (batch, 100)   ← predicted clean window
```

The 6-dimensional per-step input is:
```
x_seq[t] = [ noisy_val_t, C1, C2, C3, C4, σ ]
```
(C and σ are repeated identically at every step.)

Weights are initialised with **orthogonal initialisation** for the recurrent matrices
and Xavier uniform for input matrices to improve gradient flow.

---

## Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Hidden size | 64 | Sufficient capacity for denoising; weight-sharing across 100 steps |
| Num layers | 2 | Stacked layers add depth; second layer learns higher-order patterns |
| Bidirectional | True | Each output step sees full past + future context |
| LayerNorm | after BiRNN output | Prevents activation drift over 100 steps |
| Optimizer | Adam | lr=1e-3, weight\_decay=1e-4 |
| Loss | MSELoss | Denoising regression task |
| Gradient clip | 1.0 | Prevents exploding gradients from BPTT |

---

## Parameter Count

```
Layer 0 forward:   (6+64)×64  + 2×64  = 4,608
Layer 0 backward:  (6+64)×64  + 2×64  = 4,608
Layer 1 forward:   (128+64)×64 + 2×64 = 12,416
Layer 1 backward:  (128+64)×64 + 2×64 = 12,416
LayerNorm(128):    128×2               = 256
Linear(128→1):     128 + 1             = 129
Total:             ≈ 34,433 parameters
```

---

## Test Set Results

| Metric | Value |
|--------|-------|
| MSE | **0.005266** ← best overall |
| MAE | **0.045079** |
| Pearson r | **0.9609** |

**Per-frequency MSE:**

| Frequency | MSE |
|-----------|-----|
| 1 Hz | 0.004416 |
| 2 Hz | 0.005789 |
| 5 Hz | 0.005111 |
| 7 Hz | 0.005709 |

RNN achieves the lowest MSE at all four frequencies.

---

## Observations

- **2-layer stacking:** The second recurrent layer builds higher-order temporal
  abstractions, improving denoising over a single-layer BiRNN.
- **Bidirectionality:** Each output step sees full past + future context — important
  for sinusoidal denoising since the waveform shape extends in both directions.
- **Orthogonal init** preserves gradient norms; avoids early vanishing through 100 steps.
- **Winner at 100 epochs:** RNN's ~34K parameters converge faster than LSTM's ~535K,
  making RNN the best model within the 100-epoch CPU training budget.
