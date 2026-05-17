# PRD — Vanilla RNN

## Architecture

```
Input: (batch, SEQ_LEN, 1)
  └─ RNN(input=1, hidden=64, layers=2, tanh activation)
     └─ Take last hidden state h_T: (batch, 64)
        └─ Dropout(0.3)
           └─ Linear(64 → 4)  ← logits
```

## Hyperparameters

| Parameter    | Value | Reason |
|--------------|-------|--------|
| Hidden size  | 64    | Balanced capacity / compute |
| Num layers   | 2     | Stacked RNN improves feature extraction |
| Nonlinearity | tanh  | Standard RNN activation |
| Dropout      | 0.3   | Applied between stacked layers |
| Gradient clip| 1.0   | Prevents exploding gradients |

## Vanishing Gradient Problem

The standard RNN updates its hidden state as:
```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

Gradients flow back through time by multiplying `∂h_t/∂h_{t-1}` at each step. When this Jacobian has eigenvalues < 1, gradients **vanish exponentially** with sequence length.

For **low frequencies (1 Hz, 2 Hz)**, the discriminative pattern spans hundreds of samples. The RNN cannot retain this information — the hidden state effectively "forgets" the beginning of the window by the time it reaches step 200.

**High frequencies (5 Hz, 7 Hz)** complete multiple periods within 20-50 steps, so the RNN only needs short-term memory → it succeeds.

## Expected Behavior

| Frequency | Expected RNN performance |
|-----------|-------------------------|
| S7 (7 Hz) | High accuracy (≥ 90%)   |
| S5 (5 Hz) | Good accuracy (≥ 85%)   |
| S2 (2 Hz) | Moderate (60-80%)       |
| S1 (1 Hz) | Poor (40-65%)           |

This per-frequency degradation is visible in the confusion matrix.

## Parameter Count

`(1+64)×64×2_layers × 2 (input+hidden) ≈ 16 800 parameters`
