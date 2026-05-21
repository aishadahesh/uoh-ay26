# PRD — Fully Connected Network (FC)

## Role

FC is the **baseline model**. It processes the entire noisy window at once as a flat
vector, using the explicit frequency selector `C` and noise hint `σ` to denoise the signal.

Because FC has access to all 100 samples simultaneously, it can learn **one denoising filter
per (frequency, noise level) combination** — a nearly-linear regression that is very
efficient for stationary sinusoidal components.

---

## Architecture

```
x_flat: (batch, 105)
   └─ Linear(105 → 16) + ReLU
         └─ Linear(16 → 100)

Output: y_hat (batch, 100)   ← predicted clean window
```

The 105-dimensional input is:
```
x_flat = [ noisy_window(100) | one_hot_C(4) | sigma(1) ]
```

---

## Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Hidden size | 16 | Small capacity sufficient for simple denoising; avoids overfitting on 7K train set |
| Optimizer | Adam | lr=1e-3, weight\_decay=1e-4 |
| Loss | MSELoss | Denoising regression task |
| Scheduler | ReduceLROnPlateau | patience=5, factor=0.5 |
| Early stop patience | 15 epochs | |
| Gradient clip | 1.0 | Stability |

---

## Parameter Count

```
Linear(105 → 16):  105 × 16 + 16 = 1,696
Linear(16 → 100):  16 × 100 + 100 = 1,700
Total:             ≈ 3,396 parameters
```

---

## Test Set Results

| Metric | Value |
|--------|-------|
| MSE | 0.007321 |
| MAE | 0.052652 |
| Pearson r | 0.9572 |

**Per-frequency MSE:**

| Frequency | MSE |
|-----------|-----|
| 1 Hz | 0.006636 |
| 2 Hz | 0.007548 |
| 5 Hz | 0.007495 |
| 7 Hz | 0.007580 |

FC performs consistently across all four frequencies. The slight advantage at 1 Hz is because
the near-flat shape at 10% of the period is easier to fit with a linear matched filter.

---

## Observations

- **Full context at once.** FC processes all 100 noisy samples simultaneously; there is
  no sequential error compounding.
- **Explicit conditioning.** `C` and `σ` are concatenated directly so the network
  knows exactly which frequency to expect and how strong the noise is.
- **Efficient matched filter.** For stationary sinusoids the optimal filter is a
  linear operation. FC approximates this with a single hidden layer of 16 units.
- **Limitation.** No temporal ordering — treats the 100 samples as an exchangeable set.
  Cannot adapt to non-stationary signals or very long context windows.

