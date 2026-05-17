# PLAN — Implementation Plan

## Objective
Signal **reconstruction** (denoising): given a 10-sample noisy window + condition (C, sigma),
predict the 10-sample clean window. Loss = **MSE**.

## Data Flow
```
make_example()
  → (C, sigma, noisy_window, clean_window)
  → x_flat [15]  for FC
  → x_seq  [10,6] for RNN/LSTM
  → y      [10]   target
```

## Phases

### Phase 1 — Dataset (`src/data_generator.py`)
- [x] Constants: FREQUENCIES=[1,2,5,7], SAMPLE_RATE=1000, DURATION=10, CONTEXT_WINDOW=10
- [x] `one_hot(idx, n)` → float32 array
- [x] `generate_clean_signal(freq, A, phi)` → shape (10000,)
- [x] `add_gaussian_noise(clean, A, sigma)` → noisy copy
- [x] `make_example()` → (C, sigma, noisy_window[10], clean_window[10])
- [x] `SignalReconstructionDataset`: x_flat[15], x_seq[10,6], y[10]
- [x] `get_dataloaders()` → 70/15/15 train/val/test split

### Phase 2 — Models (`src/models.py`)
- [x] `FCNet`: Linear(15,64)→ReLU→Linear(64,64)→ReLU→Linear(64,10)
- [x] `RNNNet`: RNN(6,64)→Linear(64,1)@each_step→squeeze → [batch,10]
- [x] `LSTMNet`: LSTM(6,64)→Linear(64,1)@each_step→squeeze → [batch,10]
- [x] All models share signature: `forward(x_flat, x_seq)` → [batch,10]

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
- [x] `plot_window_example()` → window_example.png (10-sample window)
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
- [ ] README.md: full lab report with plots and analysis tables
- [x] pyproject.toml: all dependencies
- [x] .gitignore

## Key Design Decisions

1. **CONTEXT_WINDOW = 10** — per assignment spec: 10 samples per training example.
2. **FC input [batch,15]** — flat: noisy(10) + one-hot C(4) + sigma(1).
3. **RNN/LSTM input [batch,10,6]** — at each step: noisy_val + C1,C2,C3,C4 + sigma.
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

