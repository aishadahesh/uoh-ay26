# PRD — Fully Connected Network (FC)

## Role

FC is the **baseline model**. It processes the entire mixed-signal window at once as a flat
vector, using the explicit frequency selector `C` and noise hint `σ` to decide which component
to reconstruct.

Because FC has access to all 100 samples simultaneously, it can learn **one matched filter
per (frequency, noise level) combination** — a nearly-linear regression that is very
efficient for stationary sinusoidal components.

---

## Architecture

```
x_flat: (batch, 105)
   └─ Linear(105 → 64) + ReLU
         └─ Linear(64 → 100)

Output: y_hat (batch, 100)   ← predicted clean component window
```

The 105-dimensional input is:
```
x_flat = [ mixed_window(100) | one_hot_C(4) | sigma(1) ]
```

---

## Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Hidden size | 64 | Balanced capacity; same for all models |
| Optimizer | Adam | lr=1e-3, weight\_decay=1e-4 |
| Loss | MSELoss | Reconstruction task (not classification) |
| Scheduler | ReduceLROnPlateau | patience=5, factor=0.5 |
| Early stop patience | 15 epochs | |
| Gradient clip | 1.0 | Stability |

---

## Parameter Count

```
Linear(105 → 64):  105 × 64 + 64 = 6,784
Linear(64 → 100):  64 × 100 + 100 = 6,500
Total:             ≈ 13,284 parameters
```

---

## Expected Behavior

- **High-frequency components (7 Hz):** FC wins decisively (test MSE 0.186).
  70% of the oscillation period is visible in 100 samples; the amplitude/phase pattern
  is unambiguous as a flat feature vector.
- **Low-frequency components (1 Hz):** FC is slightly weaker (test MSE 0.298) because
  the 100-sample window covers only 10% of one period — the waveform looks nearly
  linear, making phase estimation harder.
- **Overall test MSE:** 0.2942 (best of the three models at this training scale).

---

## Why FC Beats Sequential Models Here

1. **Full context at once.** FC processes all 100 mixed samples simultaneously; there is
   no sequential error compounding.
2. **Explicit conditioning.** `C` and `σ` are concatenated directly, telling the network
   exactly which component to extract. FC only needs to learn the denoising filter, not
   the frequency selector.
3. **Efficient matched filter.** For stationary sinusoids, the optimal filter is a
   linear operation (inner product with the target sinusoid shape). FC can approximate
   this with a single hidden layer of 64 units.

At larger window sizes (≥ 500 samples), sequential models would begin to win because the
long context would reward temporal memory.

