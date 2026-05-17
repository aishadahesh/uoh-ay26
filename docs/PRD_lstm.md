# PRD — LSTM (Long Short-Term Memory)

## Architecture

```
Input: (batch, SEQ_LEN, 1)
  └─ LSTM(input=1, hidden=64, layers=2)
     └─ Take last hidden state h_T: (batch, 64)
        └─ Dropout(0.3)
           └─ Linear(64 → 4)  ← logits
```

## LSTM Cell Equations

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   — what to erase from cell state
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   — what new info to write
Cell update:  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell state:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t      — long-term memory
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(C_t)                  — short-term output
```

## Why LSTM Fixes RNN's Problem

The **cell state** `C_t` acts as a "conveyor belt" that can carry information across hundreds of time steps with minimal gradient attenuation. The forget gate can keep `f_t ≈ 1`, meaning `C_t ≈ C_{t-1}` — a direct gradient path backward in time with no vanishing.

This allows LSTM to:
- Detect **low-frequency** signals (1 Hz) by remembering the signal trend over 1000 samples.
- Continue to detect **high-frequency** signals (7 Hz) via normal sequential processing.

## Hyperparameters

| Parameter   | Value | Reason |
|-------------|-------|--------|
| Hidden size | 64    | Same as RNN for fair comparison |
| Num layers  | 2     | Stacked LSTM |
| Dropout     | 0.3   | Between stacked layers |
| Gradient clip | 1.0 | Stability |

## Expected Behavior

| Frequency | Expected LSTM performance |
|-----------|-------------------------|
| S7 (7 Hz) | High accuracy (≥ 97%)   |
| S5 (5 Hz) | High accuracy (≥ 97%)   |
| S2 (2 Hz) | High accuracy (≥ 95%)   |
| S1 (1 Hz) | High accuracy (≥ 90%)   |

## Parameter Count

LSTM has 4× the parameters of RNN (4 gates):
`4 × (1+64)×64 × 2_layers ≈ 66 560 parameters`
