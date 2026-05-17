# PRD — Fully Connected Network (FC)

## Architecture

```
Input: (batch, SEQ_LEN, 1)
  └─ Flatten → (batch, SEQ_LEN)
     └─ Linear(SEQ_LEN → 256) + ReLU + Dropout(0.3)
        └─ Linear(256 → 128) + ReLU + Dropout(0.3)
           └─ Linear(128 → 64) + ReLU + Dropout(0.3)
              └─ Linear(64 → 4)   ← logits
```

## Hyperparameters

| Parameter    | Value      | Reason |
|--------------|------------|--------|
| Hidden sizes | [256,128,64] | Enough capacity to learn frequency patterns |
| Dropout      | 0.3        | Regularization against overfitting |
| Optimizer    | Adam       | lr=1e-3, weight_decay=1e-4 |
| Loss         | CrossEntropy | Multi-class classification |
| Epochs       | 40         | With early-stopping via ReduceLROnPlateau |

## Expected Behavior

- FC has **no temporal awareness**; it treats the entire window as a bag of features.
- It works well at **high SNR** because the frequency pattern is learnable from the magnitude spectrum encoded implicitly in the weights.
- At high noise it degrades more than LSTM because it cannot leverage sequential structure.
- It is the **baseline** for comparison.

## Parameter Count

With SEQ_LEN=200: `200×256 + 256×128 + 128×64 + 64×4 ≈ 85 000 parameters`
