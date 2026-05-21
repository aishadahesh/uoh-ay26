# PLAN — Implementation Plan

## Objective
Signal **reconstruction** (denoising): given a 100-sample noisy window + condition (C, sigma),
predict the 100-sample clean window. Loss = **MSE**.

## Data Flow
```
make_example()
  → (C, sigma, noisy_window, clean_window)
  → x_flat [105]   for FC          ([window(100) | C(4) | sigma(1)])
  → x_seq  [100,6] for RNN/LSTM    (per step: [val, C1,C2,C3,C4,sigma])
  → y      [100]   target
```

## Phases

### Phase 1 — Dataset (`src/data_generator.py`)
- [x] Constants: FREQUENCIES=[1,2,5,7], SAMPLE_RATE=1000, DURATION=10, CONTEXT_WINDOW=100
- [x] `one_hot(idx, n)` → float32 array
- [x] `generate_clean_signal(freq, A, phi)` → shape (10000,)
- [x] `add_gaussian_noise(clean, A, sigma)` → noisy copy
- [x] `make_example()` → (C, sigma, noisy_window[100], clean_window[100])
- [x] `SignalReconstructionDataset`: x_flat[105], x_seq[100,6], y[100]
- [x] `get_dataloaders()` → 70/15/15 train/val/test split

### Phase 2 — Models (`src/models.py`)
- [x] `FCNet`: Linear(105,16)→ReLU→Linear(16,100)  (~3,400 params)
- [x] `RNNNet`: BiRNN(input=6, hidden=64, layers=2)→LayerNorm(128)→Linear(128,1)@each_step  (~34,400 params)
- [x] `LSTMNet`: BiLSTM(input=6, hidden=128, layers=2)→LayerNorm(256)→Linear(256,1)@each_step  (~535,000 params)
- [x] All models share signature: `forward(x_flat, x_seq)` → [batch,100]

### Phase 3 — Training (`src/train.py`)
- [x] MSELoss criterion
- [x] Adam optimizer (lr=1e-3, weight_decay=1e-4)
- [x] ReduceLROnPlateau scheduler (patience=5)
- [x] Gradient clipping (max_norm=1.0)
- [x] Best checkpoint saved to `results/<name>_best.pt`
- [x] Returns `{"train_loss": [...], "val_loss": [...]}`

### Phase 4 — Evaluation (`src/evaluate.py`)
- [x] `get_predictions()` → y_true, y_pred, c_idx, sigmas
- [x] `evaluate_model()` → overall MSE, MAE, mean Pearson correlation
- [x] `mse_per_frequency()` → {freq_label: mse}
- [x] `mse_per_noise_level()` → {sigma: mse}
- [x] `noise_sweep()` → list of {model, noise_level, mse}
- [x] `save_metrics_csv()` → results/metrics.csv

### Phase 5 — Plots (`src/plots.py`)
- [x] `plot_signals()` → signals.png (clean vs noisy, all 4 freqs)
- [x] `plot_window_example()` → window_example.png (100-sample window)
- [x] `plot_training_loss()` → training_loss.png (MSE curves)
- [x] `plot_prediction_vs_true()` → prediction_vs_true.png
- [x] `plot_mse_per_frequency()` → mse_per_frequency.png
- [x] `plot_noise_vs_mse()` → noise_vs_mse.png

### Phase 6 — Orchestration (`src/main.py`)
- [x] CLI with argparse: --model, --epochs, --n-samples, --batch-size, --lr, --skip-sweep
- [x] Full end-to-end pipeline
- [x] Device detection (CPU/CUDA)
- [x] Parameter count reporting

### Phase 7 — Tests (`tests/`)
- [x] `test_dataset.py`: one_hot, signal shapes, noise, make_example, dataset, dataloaders
- [x] `test_models.py`: output shapes, gradient flow, MSE loss, determinism

### Phase 8 — Documentation & GitHub
- [x] README.md: full lab report with plots and analysis tables
- [x] pyproject.toml: all dependencies
- [x] .gitignore

## Key Design Decisions

1. **CONTEXT_WINDOW = 100** — 100 samples @ 1000 Hz = 100 ms. Enough for 7 Hz to show ~70% of a period while keeping 1 Hz challenging (only 10% of a period).
2. **FC input [batch,105]** — flat: noisy(100) + one-hot C(4) + sigma(1).
3. **RNN/LSTM input [batch,100,6]** — at each step: noisy_val + C1,C2,C3,C4 + sigma.
4. **MSELoss** — required by the assignment for regression/denoising.
5. **Gaussian noise** — physically motivated (measurement error), easiest to tune via sigma.
6. **Amplitude A ∈ Uniform(0.7, 1.3)** — prevents models from memorizing a fixed amplitude.
7. **Phase φ ∈ Uniform(0, 2π)** — prevents models from memorizing a fixed phase.
8. **Gradient clipping** — critical for RNN stability with default tanh activation.
9. **ReduceLROnPlateau** — adapts learning rate when validation MSE stagnates.

## Expected Results
- At sigma=0: all models should converge to very low MSE (<1e-3).
- At sigma=0.20: MSE increases; LSTM expected to recover better than RNN and FC.
- 1 Hz frequency: hardest to reconstruct (10 samples = 1% of one period).
- 7 Hz frequency: easier (10 samples ≈ 70% of one period → more signal visible).

