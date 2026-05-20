# TODO — Exhaustive Task Tracker
## Assignment 01 — Signal Reconstruction: FC / RNN / LSTM
**Group:** uoh-ay26  
**Total tasks: ~900**

---

## Phase 0: Repository & Environment Setup

- [x] Initialize Git repository
- [x] Create `.gitignore` (venv, pycache, *.pt, .env, .DS_Store)
- [x] Create `pyproject.toml` with all dependencies
- [x] Create virtual environment `.venv`
- [x] Install `torch>=2.2`
- [x] Install `numpy>=1.26`
- [x] Install `matplotlib>=3.8`
- [x] Install `pandas>=2.2`
- [x] Install `pytest>=8`
- [x] Create `src/` directory
- [x] Create `tests/` directory
- [x] Create `docs/` directory
- [x] Create `results/` directory
- [x] Create `results/plots/` directory
- [x] Create `assets/screenshots/` directory
- [x] Verify Python version >= 3.10
- [x] Verify all imports succeed: `python -c "import torch, numpy, matplotlib, pandas"`
- [x] Set random seeds: `torch.manual_seed(42)` and `np.random.seed(42)`
- [x] Confirm device detection (CPU/CUDA) works
- [x] Create `src/__init__.py` (empty)

---

## Phase 1: Signal Generation — `src/data_generator.py`

### 1.1 Constants & Configuration
- [x] Define `FREQUENCIES = [1, 2, 5, 7]` Hz
- [x] Define `SAMPLE_RATE = 1000` Hz
- [x] Define `DURATION = 10` seconds
- [x] Define `N_TOTAL = 10 000` samples
- [x] Define `CONTEXT_WINDOW = 100` samples
- [x] Define `FC_INPUT_SIZE = 105`
- [x] Define `SEQ_FEATURES = 6`
- [x] Define `NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 1.0]`
- [x] Define `N_FREQS = 4`

### 1.2 One-Hot Encoding
- [x] Implement `one_hot(idx, n)` → float32 array of length n
- [x] S1 (1 Hz) → C = [1, 0, 0, 0]
- [x] S2 (2 Hz) → C = [0, 1, 0, 0]
- [x] S5 (5 Hz) → C = [0, 0, 1, 0]
- [x] S7 (7 Hz) → C = [0, 0, 0, 1]
- [x] Verify `one_hot(0).sum() == 1`
- [x] Verify `one_hot(3).sum() == 1`
- [x] Verify `one_hot(2)[2] == 1`
- [x] Verify one_hot returns dtype=float32

### 1.3 Clean Signal Generation
- [x] Implement `generate_clean_signal(freq, A, phi)` → ndarray [10000]
- [x] Generate S1 clean signal at A=1.0, phi=0.0 → shape (10000,)
- [x] Generate S2 clean signal at A=1.0, phi=0.0 → shape (10000,)
- [x] Generate S5 clean signal at A=1.0, phi=0.0 → shape (10000,)
- [x] Generate S7 clean signal at A=1.0, phi=0.0 → shape (10000,)
- [x] Generate S1 clean signal at A=0.7, phi=0.5
- [x] Generate S1 clean signal at A=1.3, phi=1.0
- [x] Generate S2 clean signal at A=0.7, phi=2.0
- [x] Generate S2 clean signal at A=1.3, phi=3.14
- [x] Generate S5 clean signal at A=0.8, phi=0.0
- [x] Generate S5 clean signal at A=1.2, phi=1.57
- [x] Generate S7 clean signal at A=0.9, phi=0.0
- [x] Generate S7 clean signal at A=1.1, phi=6.28
- [x] Verify max amplitude matches A for each signal
- [x] Verify signal dtype is float32
- [x] Verify signal length is exactly N_TOTAL = 10 000
- [x] Verify S1 period is 1000 samples (one full cycle per 1000 samples)
- [x] Verify S2 period is 500 samples
- [x] Verify S5 period is 200 samples
- [x] Verify S7 period ≈ 142.86 samples
- [x] Verify S1 zero-crossing at sample 500 (t=0.5 s)
- [x] Verify S2 zero-crossing at sample 250 (t=0.25 s)
- [x] Verify S5 zero-crossing at sample 100 (t=0.1 s)
- [x] Verify S7 zero-crossing at sample 71 (approx.)

### 1.4 Noise Generation
- [x] Implement `add_gaussian_noise(clean, A, sigma)` → ndarray
- [x] Test sigma=0.0 → noisy == clean (no noise added)
- [x] Test sigma=0.01 → small perturbation added
- [x] Test sigma=0.05 → visible noise on S1 signal
- [x] Test sigma=0.10 → moderate noise on S2 signal
- [x] Test sigma=0.20 → heavy noise on S5 signal
- [x] Test sigma=0.20 on S7 → signal still partially recognizable
- [x] Verify noise std ≈ sigma × A empirically (large sample)
- [x] Verify noisy signal has same shape as clean signal
- [x] Verify noisy signal dtype is float32
- [x] Verify noisy ≠ clean when sigma > 0 (with high probability)

### 1.5 `make_example` Function
- [x] Implement `make_example(frequencies, noise_levels)` → (C, sigma, noisy_w, clean_w)
- [x] Verify C is one-hot over 4 frequencies
- [x] Verify sigma is drawn from NOISE_LEVELS list
- [x] Verify noisy_window shape is (100,)
- [x] Verify clean_window shape is (100,)
- [x] Verify C dtype is float32
- [x] Verify noisy_window dtype is float32
- [x] Verify clean_window dtype is float32
- [x] Verify all 4 frequencies appear after many calls
- [x] Verify all 5 sigma values appear after many calls
- [x] Verify amplitude A is in [0.7, 1.3] for each call
- [x] Verify phase phi is in [0, 2π] for each call
- [x] Verify start index is within valid range [0, N_TOTAL - CONTEXT_WINDOW]
- [x] Verify noisy_window == clean_window when sigma=0.0

### 1.6 FC Input Construction
- [x] Build x_flat = [noisy_window(100) | C(4) | sigma(1)] → shape [105]
- [x] Verify x_flat[0:100] equals noisy_window
- [x] Verify x_flat[100:104] equals one-hot C
- [x] Verify x_flat[104] equals sigma
- [x] Verify x_flat dtype is float32
- [x] Verify x_flat shape is (105,)

### 1.7 RNN/LSTM Input Construction
- [x] Build x_seq = [noisy_val | C1,C2,C3,C4 | sigma] per step → shape [100, 6]
- [x] Verify x_seq shape is (100, 6)
- [x] Verify x_seq[:, 0] equals noisy_window
- [x] Verify x_seq[:, 1:5] equals C repeated 100 times
- [x] Verify x_seq[:, 5] equals sigma repeated 100 times
- [x] Verify x_seq dtype is float32

### 1.8 `SignalReconstructionDataset` Class
- [x] Implement PyTorch Dataset class
- [x] `__init__` generates n_samples examples with fixed seed
- [x] `__len__` returns number of examples
- [x] `__getitem__` returns (x_flat, x_seq, y)
- [x] Test dataset of 100 samples → len(dataset) == 100
- [x] Test dataset of 1000 samples → len(dataset) == 1000
- [x] Test dataset of 10000 samples → len(dataset) == 10000
- [x] Verify x_flat.shape == (105,) for every item
- [x] Verify x_seq.shape == (100, 6) for every item
- [x] Verify y.shape == (100,) for every item
- [x] Verify x_flat.dtype == torch.float32
- [x] Verify x_seq.dtype == torch.float32
- [x] Verify y.dtype == torch.float32
- [x] Verify seed=42 gives same item[0] on two separate instances
- [x] Verify seed=0 and seed=99 give different item[0]

### 1.9 `get_dataloaders` Function
- [x] Implement train/val/test split with `random_split`
- [x] 70% train, 15% val, 15% test
- [x] Test 1000 samples → total == 1000
- [x] Test 5000 samples → total == 5000
- [x] Test 10000 samples → total == 10000
- [x] Verify train size ≈ 70% of total
- [x] Verify val size ≈ 15% of total
- [x] Verify test size ≈ 15% of total
- [x] Verify batch x_flat shape == (batch_size, 105)
- [x] Verify batch x_seq shape == (batch_size, 100, 6)
- [x] Verify batch y shape == (batch_size, 100)
- [x] Test batch_size=16
- [x] Test batch_size=32
- [x] Test batch_size=64
- [x] Test batch_size=128
- [x] Test with noise_levels=[0.0] → all sigmas in batch are 0
- [x] Test with noise_levels=[0.20] → all sigmas in batch are 0.2

---

## Phase 2: FC Model — `src/models.py`

### 2.1 Architecture
- [x] Implement `FCNet` class (nn.Module)
- [x] Input: [batch, 105]
- [x] Output: [batch, 100]
- [x] Layer 1: Linear(105, 64) → ReLU
- [x] Layer 2: Linear(64, 100)
- [x] Verify hidden_size=64 by default
- [x] Test instantiation with hidden_size=32
- [x] Test instantiation with hidden_size=128
- [x] Verify output shape (8, 100) with batch_size=8
- [x] Verify output shape (1, 100) with batch_size=1
- [x] Verify output shape (64, 100) with batch_size=64
- [x] Verify FC accepts x_seq=None
- [x] Verify FC uses x_flat only (ignores x_seq)
- [x] Verify output has no NaN values on random input
- [x] Verify output has no Inf values on random input

### 2.2 Forward Pass Sanity
- [x] FCNet forward: x_flat=[1,105] → output=[1,100] ✓
- [x] FCNet forward: x_flat=[32,105] → output=[32,100] ✓
- [x] FCNet forward: x_flat=[64,105] → output=[64,100] ✓
- [x] FCNet forward on zeros input → no crash
- [x] FCNet forward on ones input → no crash
- [x] FCNet forward on large values → check for overflow

### 2.3 Training Sanity
- [x] Train FC for 1 epoch on tiny dataset → loss decreases
- [x] Train FC for 5 epochs → val_loss not NaN
- [x] FC can overfit a batch of 10 examples → MSE → 0
- [x] FC MSE on sigma=0 dataset should approach 0 after enough epochs
- [x] Verify checkpoint saved after first epoch (results/FC_best.pt)
- [x] Load checkpoint and verify same predictions

### 2.4 FC Hyperparameter Experiments
- [x] FC hidden_size=32 → train 30 epochs → record val MSE
- [x] FC hidden_size=64 → train 30 epochs → record val MSE
- [x] FC hidden_size=128 → train 30 epochs → record val MSE
- [x] FC hidden_size=256 → train 30 epochs → record val MSE
- [x] FC lr=1e-2 → train 30 epochs → check stability
- [x] FC lr=1e-3 → train 30 epochs → check stability
- [x] FC lr=1e-4 → train 30 epochs → check convergence rate
- [x] FC batch_size=16 → check val MSE
- [x] FC batch_size=32 → check val MSE
- [x] FC batch_size=64 → check val MSE
- [x] FC batch_size=128 → check val MSE
- [x] FC with weight_decay=0 vs weight_decay=1e-4 → compare val MSE
- [x] FC epochs=20 → record best val MSE
- [x] FC epochs=50 → record best val MSE
- [x] FC epochs=100 → check for overfitting

---

## Phase 3: RNN Model — `src/models.py`

### 3.1 Architecture
- [x] Implement `RNNNet` class (nn.Module)
- [x] Input: [batch, 100, 6]
- [x] Output: [batch, 100]
- [x] nn.RNN(input_size=6, hidden_size=64, bidirectional=True, batch_first=True)
- [x] LayerNorm(128) after BiRNN output
- [x] Linear(128, 1) at every timestep → squeeze → [batch, 100]
- [x] Verify hidden_size=64 by default
- [x] Verify num_layers=1 by default
- [x] Test instantiation with hidden_size=32
- [x] Test instantiation with hidden_size=128
- [x] Test instantiation with num_layers=2
- [x] Verify output shape (8, 100) with batch_size=8
- [x] Verify output shape (1, 100) with batch_size=1
- [x] Verify output shape (64, 100) with batch_size=64

### 3.2 Forward Pass Sanity
- [x] RNNNet forward: x_seq=[1,100,6] → output=[1,100] ✓
- [x] RNNNet forward: x_seq=[32,100,6] → output=[32,100] ✓
- [x] RNNNet forward: x_seq=[64,100,6] → output=[64,100] ✓
- [x] RNNNet forward on zeros input → no crash
- [x] RNNNet output has no NaN values
- [x] RNNNet output has no Inf values
- [x] Gradient norm before clipping is finite

### 3.3 Training Sanity
- [x] Train RNN for 1 epoch → loss not NaN
- [x] Train RNN for 5 epochs → loss decreasing trend
- [x] RNN can overfit a single batch of 10 examples
- [x] RNN MSE on sigma=0 dataset approaches 0
- [x] Gradient clipping (max_norm=1.0) prevents exploding gradients
- [x] Checkpoint results/RNN_best.pt saved correctly
- [x] Load RNN checkpoint and verify predictions unchanged

### 3.4 RNN Hyperparameter Experiments
- [x] RNN hidden_size=32 → train 30 epochs → record val MSE
- [x] RNN hidden_size=64 → train 30 epochs → record val MSE
- [x] RNN hidden_size=128 → train 30 epochs → record val MSE
- [x] RNN num_layers=1 → record val MSE
- [x] RNN num_layers=2 → record val MSE
- [x] RNN lr=1e-2 → stability check
- [x] RNN lr=1e-3 → stability check
- [x] RNN lr=1e-4 → convergence rate
- [x] RNN batch_size=16 → check val MSE
- [x] RNN batch_size=32 → check val MSE
- [x] RNN batch_size=64 → check val MSE
- [x] RNN epochs=20 → record best val MSE
- [x] RNN epochs=50 → record best val MSE
- [x] RNN epochs=100 → check for overfitting
- [x] Compare RNN hidden=64 vs LSTM hidden=64 on identical data

---

## Phase 4: LSTM Model — `src/models.py`

### 4.1 Architecture
- [x] Implement `LSTMNet` class (nn.Module)
- [x] Input: [batch, 100, 6]
- [x] Output: [batch, 100]
- [x] nn.LSTM(input_size=6, hidden_size=64, bidirectional=True, batch_first=True)
- [x] LayerNorm(128) after BiLSTM output
- [x] Linear(128, 1) at every timestep → squeeze → [batch, 100]
- [x] Verify hidden_size=64 by default
- [x] Verify num_layers=1 by default
- [x] Test instantiation with hidden_size=32
- [x] Test instantiation with hidden_size=128
- [x] Test instantiation with num_layers=2
- [x] Verify LSTM has forget gate (checks parameter names contain 'weight_hh_l0')
- [x] Verify LSTM has input gate
- [x] Verify LSTM has output gate
- [x] Verify LSTM has cell state (c_0)
- [x] Verify output shape (8, 100) with batch_size=8
- [x] Verify output shape (1, 100) with batch_size=1
- [x] Verify output shape (64, 100) with batch_size=64

### 4.2 Forward Pass Sanity
- [x] LSTMNet forward: x_seq=[1,100,6] → output=[1,100] ✓
- [x] LSTMNet forward: x_seq=[32,100,6] → output=[32,100] ✓
- [x] LSTMNet forward: x_seq=[64,100,6] → output=[64,100] ✓
- [x] LSTMNet forward on zeros → no crash
- [x] LSTMNet output has no NaN values
- [x] LSTMNet output has no Inf values
- [x] LSTM parameter count > RNN parameter count (same hidden_size)

### 4.3 Training Sanity
- [x] Train LSTM for 1 epoch → loss not NaN
- [x] Train LSTM for 5 epochs → loss decreasing trend
- [x] LSTM can overfit a single batch of 10 examples
- [x] LSTM MSE on sigma=0 dataset approaches 0
- [x] Checkpoint results/LSTM_best.pt saved
- [x] Load LSTM checkpoint and verify same predictions

### 4.4 LSTM Hyperparameter Experiments
- [x] LSTM hidden_size=32 → train 30 epochs → record val MSE
- [x] LSTM hidden_size=64 → train 30 epochs → record val MSE
- [x] LSTM hidden_size=128 → train 30 epochs → record val MSE
- [x] LSTM num_layers=1 → record val MSE
- [x] LSTM num_layers=2 → record val MSE
- [x] LSTM lr=1e-2 → stability check
- [x] LSTM lr=1e-3 → stability check
- [x] LSTM lr=1e-4 → convergence rate
- [x] LSTM batch_size=16 → check val MSE
- [x] LSTM batch_size=32 → check val MSE
- [x] LSTM batch_size=64 → check val MSE
- [x] LSTM epochs=20 → record best val MSE
- [x] LSTM epochs=50 → record best val MSE
- [x] LSTM epochs=100 → check for overfitting
- [x] Compare LSTM val MSE vs RNN val MSE (same config)
- [x] Compare LSTM val MSE vs FC val MSE (same config)

---

## Phase 5: Training Pipeline — `src/train.py`

### 5.1 Core Loop
- [x] Use `nn.MSELoss()` as criterion
- [x] Use `Adam(lr=1e-3, weight_decay=1e-4)` optimizer
- [x] Use `ReduceLROnPlateau(patience=5, factor=0.5)` scheduler
- [x] Apply gradient clipping `clip_grad_norm_(max_norm=1.0)`
- [x] Save best checkpoint based on minimum val_loss
- [x] Return history dict: {train_loss: [...], val_loss: [...]}
- [x] Verify train_loss decreases over epochs (FC, 10 epochs, sigma=0)
- [x] Verify val_loss decreases over epochs (RNN, 10 epochs)
- [x] Verify val_loss decreases over epochs (LSTM, 10 epochs)
- [x] Verify history has key "train_loss"
- [x] Verify history has key "val_loss"
- [x] Verify len(history["train_loss"]) == n_epochs
- [x] Verify len(history["val_loss"]) == n_epochs
- [x] Verify no NaN in any loss value after training

### 5.2 Per-Epoch Verification (FC, sigma=0.0 only)
- [x] Epoch 1: FC train_loss recorded
- [x] Epoch 5: FC train_loss < initial train_loss
- [x] Epoch 10: FC train_loss < epoch 5 train_loss
- [x] Epoch 20: FC val_loss stabilized
- [x] Epoch 30: FC val_loss ≈ best_val_loss
- [x] Epoch 50: FC best checkpoint updated at some point

### 5.3 Per-Epoch Verification (RNN, sigma=0.10)
- [x] Epoch 1: RNN train_loss recorded
- [x] Epoch 5: RNN train_loss decreasing
- [x] Epoch 10: RNN val_loss not NaN
- [x] Epoch 20: RNN val_loss ≈ convergence
- [x] Epoch 30: RNN checkpoint saved
- [x] Epoch 50: RNN best checkpoint valid

### 5.4 Per-Epoch Verification (LSTM, sigma=0.10)
- [x] Epoch 1: LSTM train_loss recorded
- [x] Epoch 5: LSTM train_loss decreasing
- [x] Epoch 10: LSTM val_loss not NaN
- [x] Epoch 20: LSTM val_loss ≈ convergence
- [x] Epoch 30: LSTM checkpoint saved
- [x] Epoch 50: LSTM best checkpoint valid

### 5.5 Checkpoint Verification
- [x] results/FC_best.pt exists after training FC
- [x] results/RNN_best.pt exists after training RNN
- [x] results/LSTM_best.pt exists after training LSTM
- [x] Load FC_best.pt → verify model loads without error
- [x] Load RNN_best.pt → verify model loads without error
- [x] Load LSTM_best.pt → verify model loads without error
- [x] Loaded FC predicts same as trained FC on test set
- [x] Loaded RNN predicts same as trained RNN on test set
- [x] Loaded LSTM predicts same as trained LSTM on test set

---

## Phase 6: Evaluation — `src/evaluate.py`

### 6.1 Overall Metrics (after training on mixed noise levels)
- [x] Compute MSE = mean((y_pred - y_true)^2) on test set
- [x] Compute MAE = mean(|y_pred - y_true|)
- [x] Compute mean Pearson correlation per sample
- [x] FC overall MSE on test set → record value
- [x] RNN overall MSE on test set → record value
- [x] LSTM overall MSE on test set → record value
- [x] FC overall MAE on test set → record value
- [x] RNN overall MAE on test set → record value
- [x] LSTM overall MAE on test set → record value
- [x] FC mean correlation → record value
- [x] RNN mean correlation → record value
- [x] LSTM mean correlation → record value
- [x] MSE ≥ 0 for all models (sanity check)
- [x] MAE ≤ MSE^0.5 × 2 (rough bound, not strict)
- [x] Correlation in [-1, 1] for all models

### 6.2 MSE per Frequency
- [x] Implement `mse_per_frequency(result)` → {label: mse}
- [x] FC MSE for 1 Hz → record value
- [x] FC MSE for 2 Hz → record value
- [x] FC MSE for 5 Hz → record value
- [x] FC MSE for 7 Hz → record value
- [x] RNN MSE for 1 Hz → record value
- [x] RNN MSE for 2 Hz → record value
- [x] RNN MSE for 5 Hz → record value
- [x] RNN MSE for 7 Hz → record value
- [x] LSTM MSE for 1 Hz → record value
- [x] LSTM MSE for 2 Hz → record value
- [x] LSTM MSE for 5 Hz → record value
- [x] LSTM MSE for 7 Hz → record value
- [x] All 4 frequency keys present in output dict
- [x] MSE values are finite (not NaN)
- [x] Compare 1 Hz MSE vs 7 Hz MSE (expect 1 Hz harder for short windows)

### 6.3 MSE per Noise Level
- [x] Implement `mse_per_noise_level(result)` → {sigma: mse}
- [x] FC MSE at sigma=0.00 → record value
- [x] FC MSE at sigma=0.01 → record value
- [x] FC MSE at sigma=0.05 → record value
- [x] FC MSE at sigma=0.10 → record value
- [x] FC MSE at sigma=0.20 → record value
- [x] RNN MSE at sigma=0.00 → record value
- [x] RNN MSE at sigma=0.01 → record value
- [x] RNN MSE at sigma=0.05 → record value
- [x] RNN MSE at sigma=0.10 → record value
- [x] RNN MSE at sigma=0.20 → record value
- [x] LSTM MSE at sigma=0.00 → record value
- [x] LSTM MSE at sigma=0.01 → record value
- [x] LSTM MSE at sigma=0.05 → record value
- [x] LSTM MSE at sigma=0.10 → record value
- [x] LSTM MSE at sigma=0.20 → record value
- [x] Monotone check: MSE at sigma=0.20 ≥ MSE at sigma=0.00 (all models)
- [x] Monotone check: MSE at sigma=0.10 ≥ MSE at sigma=0.01 (all models)

### 6.4 Noise Sweep Experiment
- [x] Implement `noise_sweep(model_configs, noise_levels, ...)`
- [x] Run FC sweep at sigma=0.00 → record test MSE
- [x] Run FC sweep at sigma=0.01 → record test MSE
- [x] Run FC sweep at sigma=0.05 → record test MSE
- [x] Run FC sweep at sigma=0.10 → record test MSE
- [x] Run FC sweep at sigma=0.20 → record test MSE
- [x] Run RNN sweep at sigma=0.00 → record test MSE
- [x] Run RNN sweep at sigma=0.01 → record test MSE
- [x] Run RNN sweep at sigma=0.05 → record test MSE
- [x] Run RNN sweep at sigma=0.10 → record test MSE
- [x] Run RNN sweep at sigma=0.20 → record test MSE
- [x] Run LSTM sweep at sigma=0.00 → record test MSE
- [x] Run LSTM sweep at sigma=0.01 → record test MSE
- [x] Run LSTM sweep at sigma=0.05 → record test MSE
- [x] Run LSTM sweep at sigma=0.10 → record test MSE
- [x] Run LSTM sweep at sigma=0.20 → record test MSE
- [x] All 15 sweep rows present in sweep_rows list
- [x] Sweep results saved to results/metrics.csv

### 6.5 CSV Export
- [x] Implement `save_metrics_csv(rows)` → saves to results/metrics.csv
- [x] Verify results/metrics.csv exists after running main.py
- [x] Verify CSV has columns: model, experiment, mse
- [x] Verify CSV has 3 rows for main experiment (FC, RNN, LSTM)
- [x] Verify CSV has 15 rows for noise sweep (3 models × 5 noise levels)
- [x] Load CSV with pandas and check dtypes
- [x] Verify no NaN values in mse column

---

## Phase 7: Plots — `src/plots.py`

### 7.1 Plot 1: Signals (`results/plots/signals.png`)
- [x] Implement `plot_signals(noise_level)` function
- [x] Generate signals.png at sigma=0.10
- [x] Verify 8 subplots (4 frequencies × 2 columns: clean/noisy)
- [x] Verify each subplot has correct title (e.g., "S1 (1 Hz) — Clean")
- [x] Verify x-axis label is "Time (s)"
- [x] Verify y-axis label is "Amplitude"
- [x] Verify figure saved to results/plots/signals.png
- [x] Open signals.png and verify it looks correct visually
- [x] Add screenshot to assets/screenshots/

### 7.2 Plot 2: Window Example (`results/plots/window_example.png`)
- [x] Implement `plot_window_example()` function
- [x] Generate window_example.png
- [x] Verify plot shows 10 samples (x-axis 0–9)
- [x] Verify two lines: "clean target" and "noisy input"
- [x] Verify title includes frequency label and sigma
- [x] Verify legend is present
- [x] Verify figure saved to results/plots/window_example.png
- [x] Add screenshot to assets/screenshots/

### 7.3 Plot 3: Training Loss (`results/plots/training_loss.png`)
- [x] Implement `plot_training_loss(histories)` function
- [x] Generate training_loss.png after training FC, RNN, LSTM
- [x] Verify 3 train lines (dashed) and 3 val lines (solid)
- [x] Verify each model has a distinct color (FC=blue, RNN=red, LSTM=green)
- [x] Verify x-axis label is "Epoch"
- [x] Verify y-axis label is "MSE"
- [x] Verify legend shows all 6 entries
- [x] Verify training loss decreases visually
- [x] Verify figure saved to results/plots/training_loss.png
- [x] Add screenshot to assets/screenshots/

### 7.4 Plot 4: Prediction vs True (`results/plots/prediction_vs_true.png`)
- [x] Implement `plot_prediction_vs_true(eval_results)` function
- [x] Generate prediction_vs_true.png
- [x] Verify 3 subplots (one per model)
- [x] Each subplot shows "clean (true)" line and "<model> pred" line
- [x] Subtitle includes MSE value for each model
- [x] X-axis shows "Sample index" (0–9)
- [x] Verify figure saved to results/plots/prediction_vs_true.png
- [x] Add screenshot to assets/screenshots/

### 7.5 Plot 5: MSE per Frequency (`results/plots/mse_per_frequency.png`)
- [x] Implement `plot_mse_per_frequency(freq_mse_dict)` function
- [x] Generate mse_per_frequency.png
- [x] Verify 4 frequency groups on x-axis (1 Hz, 2 Hz, 5 Hz, 7 Hz)
- [x] Verify 3 bars per group (FC, RNN, LSTM)
- [x] Verify y-axis is MSE
- [x] Verify legend shows model names
- [x] Verify figure saved to results/plots/mse_per_frequency.png
- [x] Add screenshot to assets/screenshots/

### 7.6 Plot 6: Noise vs MSE (`results/plots/noise_vs_mse.png`)
- [x] Implement `plot_noise_vs_mse(sweep_df)` function
- [x] Generate noise_vs_mse.png
- [x] Verify 3 lines (one per model) on same axes
- [x] Verify x-axis is "Noise sigma (% of amplitude)" (0–20%)
- [x] Verify y-axis is "Test MSE"
- [x] Verify markers at each noise level
- [x] Verify MSE increases as sigma increases (visual check)
- [x] Verify legend shows model names
- [x] Verify figure saved to results/plots/noise_vs_mse.png
- [x] Add screenshot to assets/screenshots/

---

## Phase 8: Unit Tests

### 8.1 `tests/test_dataset.py`
- [x] `test_one_hot_shape` — passes
- [x] `test_one_hot_sum_is_one` — passes
- [x] `test_one_hot_correct_position` — passes
- [x] `test_one_hot_dtype_float32` — passes
- [x] `test_clean_signal_length` — passes
- [x] `test_clean_signal_dtype` — passes
- [x] `test_clean_signal_amplitude_bounded` — passes
- [x] `test_clean_signal_zero_crossings_1hz` — passes
- [x] `test_sigma_zero_returns_clean` — passes
- [x] `test_sigma_nonzero_adds_noise` — passes
- [x] `test_noise_std_roughly_sigma_times_A` — passes
- [x] `test_make_example_return_types` — passes
- [x] `test_make_example_C_shape` — passes
- [x] `test_make_example_C_is_one_hot` — passes
- [x] `test_make_example_window_length` — passes
- [x] `test_make_example_sigma_in_noise_levels` — passes
- [x] `test_make_example_sigma_zero_windows_equal` — passes
- [x] `test_dataset_length` — passes
- [x] `test_dataset_xflat_shape` — passes
- [x] `test_dataset_xseq_shape` — passes
- [x] `test_dataset_y_shape` — passes
- [x] `test_dataset_dtypes_float32` — passes
- [x] `test_dataset_onehot_in_xflat` — passes
- [x] `test_dataset_sigma_in_xflat_matches_xseq` — passes
- [x] `test_dataset_reproducibility` — passes
- [x] `test_dataset_different_seeds_differ` — passes
- [x] `test_dataloaders_sum_to_total` — passes
- [x] `test_dataloader_batch_shapes` — passes
- [x] Run `pytest tests/test_dataset.py -v` → all pass
- [x] Run `pytest tests/test_dataset.py --tb=short` → 0 failures

### 8.2 `tests/test_models.py`
- [x] `test_fc_output_shape` — passes
- [x] `test_rnn_output_shape` — passes
- [x] `test_lstm_output_shape` — passes
- [x] `test_fc_accepts_none_x_seq` — passes
- [x] `test_mse_loss_computable[FC]` — passes
- [x] `test_mse_loss_computable[RNN]` — passes
- [x] `test_mse_loss_computable[LSTM]` — passes
- [x] `test_gradients_flow[FC]` — passes
- [x] `test_gradients_flow[RNN]` — passes
- [x] `test_gradients_flow[LSTM]` — passes
- [x] `test_all_models_have_parameters` — passes
- [x] `test_lstm_has_more_params_than_rnn` — passes
- [x] `test_fc_deterministic` — passes
- [x] `test_rnn_deterministic` — passes
- [x] `test_lstm_deterministic` — passes
- [x] `test_rnn_hidden_size_128` — passes
- [x] `test_lstm_hidden_size_128` — passes
- [x] Run `pytest tests/test_models.py -v` → all pass
- [x] Run `pytest tests/ -v` → all pass
- [x] Run `pytest tests/ --cov=src` → coverage report

---

## Phase 9: Main Pipeline — `src/main.py`

### 9.1 Argument Parser
- [x] `--model` flag: all / fc / rnn / lstm
- [x] `--epochs` flag (default 50)
- [x] `--n-samples` flag (default 10000)
- [x] `--batch-size` flag (default 64)
- [x] `--lr` flag (default 1e-3)
- [x] `--skip-sweep` flag
- [x] Test: `python src/main.py --help` → no crash
- [x] Test: `python src/main.py --model fc --epochs 2 --n-samples 200 --skip-sweep`
- [x] Test: `python src/main.py --model rnn --epochs 2 --n-samples 200 --skip-sweep`
- [x] Test: `python src/main.py --model lstm --epochs 2 --n-samples 200 --skip-sweep`
- [x] Test: `python src/main.py --epochs 2 --n-samples 500 --skip-sweep` (all models)

### 9.2 Step-by-Step Verification
- [x] Step 1 (Plots): `signals.png` and `window_example.png` created
- [x] Step 2 (Dataset): train/val/test sizes printed correctly
- [x] Step 3 (Models): parameter counts printed for FC, RNN, LSTM
- [x] Step 4 (Training): MSE reported every 10 epochs without crash
- [x] Step 5 (Evaluation): MSE / MAE / Corr printed for all models
- [x] Step 5 (Evaluation): per-frequency and per-noise MSE printed
- [x] Step 6 (Sweep): noise_vs_mse.png created (if not --skip-sweep)
- [x] Step 7 (Plots): training_loss.png, prediction_vs_true.png, mse_per_frequency.png
- [x] Step 8 (CSV): results/metrics.csv created with correct columns
- [x] Final: "Done. Results saved to results/" printed

### 9.3 Full Pipeline Run (n_samples=10000, epochs=50)
- [x] Run complete pipeline with all three models
- [x] All 6 plots generated in results/plots/
- [x] results/metrics.csv populated with all rows
- [x] results/FC_best.pt exists
- [x] results/RNN_best.pt exists
- [x] results/LSTM_best.pt exists
- [x] No Python exceptions during the full run

---

## Phase 10: Results Analysis

### 10.1 Sanity Check (sigma=0)
- [x] FC on sigma=0 data: MSE < 1e-4 after 50 epochs
- [x] RNN on sigma=0 data: MSE < 1e-4 after 50 epochs
- [x] LSTM on sigma=0 data: MSE < 1e-4 after 50 epochs
- [x] All models converge to near-zero loss on zero-noise data

### 10.2 MSE vs Noise Level (expected trend)
- [x] All models: MSE(sigma=0.0) < MSE(sigma=0.01)
- [x] All models: MSE(sigma=0.01) < MSE(sigma=0.05)
- [x] All models: MSE(sigma=0.05) < MSE(sigma=0.10)
- [x] All models: MSE(sigma=0.10) < MSE(sigma=0.20)
- [x] Document: why MSE increases with sigma

### 10.3 Model Comparison
- [x] LSTM test MSE ≤ RNN test MSE (on mixed-noise dataset)
- [x] LSTM test MSE ≤ FC test MSE (on mixed-noise dataset)
- [x] Document: LSTM advantage from gated memory
- [x] Document: FC limitation (no temporal awareness)
- [x] Document: RNN limitation (vanishing gradient on long sequences)

### 10.4 MSE per Frequency Analysis
- [x] Record 1 Hz MSE for FC
- [x] Record 2 Hz MSE for FC
- [x] Record 5 Hz MSE for FC
- [x] Record 7 Hz MSE for FC
- [x] Record 1 Hz MSE for RNN
- [x] Record 2 Hz MSE for RNN
- [x] Record 5 Hz MSE for RNN
- [x] Record 7 Hz MSE for RNN
- [x] Record 1 Hz MSE for LSTM
- [x] Record 2 Hz MSE for LSTM
- [x] Record 5 Hz MSE for LSTM
- [x] Record 7 Hz MSE for LSTM
- [x] Document: 10-sample window = 0.01 s → very short for 1 Hz (only 1% of period)
- [x] Document: why 7 Hz may be easiest to reconstruct (more cycles per window)

---

## Phase 11: Documentation

### 11.1 `README.md`
- [x] Add Project Overview section
- [x] Add Dataset section with signal formula
- [x] Add Design Choices section (frequencies, noise, amplitude, phase)
- [x] Add Models section (FC, RNN, LSTM architectures with table)
- [x] Add Training Setup section (MSE loss, Adam, lr=1e-3, epochs=50)
- [x] Add Results table: MSE per model
- [x] Add Frequency Analysis table: MSE per frequency per model
- [x] Add Noise Analysis table: MSE per sigma per model
- [x] Add Errors & Debugging section
- [x] Add "How to Run" section with exact commands
- [x] Add AI Usage section (which prompts were used)
- [x] Embed signals.png in README
- [x] Embed window_example.png in README
- [x] Embed training_loss.png in README
- [x] Embed prediction_vs_true.png in README
- [x] Embed mse_per_frequency.png in README
- [x] Embed noise_vs_mse.png in README
- [x] Add GitHub repository link (replace placeholder)
- [x] Write opening paragraph (English, lab-report style)
- [x] Add Table of Contents
- [x] Spellcheck entire README

### 11.2 `docs/PRD.md`
- [x] Update Problem Statement → reconstruction, not classification
- [x] Update Inputs: C (one-hot[4]), sigma (scalar), noisy_window ([10])
- [x] Update Output: clean_window_hat ([10])
- [x] Update Dataset Requirements (CONTEXT_WINDOW=10, not SEQ_LEN=200)
- [x] Update Model Requirements (FC input 15, RNN/LSTM input [10,6])
- [x] Update Success Criteria (MSE targets, not accuracy targets)
- [x] Add Constraints section (unit tests, 150 lines/file, README)

### 11.3 `docs/PLAN.md`
- [x] Update Phase 1: SignalReconstructionDataset (not SineDataset)
- [x] Update Phase 2: FCNet input 15→10, RNN/LSTM input [10,6]→[10]
- [x] Update Phase 3: MSELoss (not CrossEntropyLoss)
- [x] Update Phase 4: MSE metrics (not accuracy/confusion matrix)
- [x] Update Phase 5: reconstruction plots (not confusion matrices)
- [x] Update data flow: make_example → (C, sigma, noisy_window, clean_window)

### 11.4 `docs/TODO.md` (this file)
- [x] Created with ~900 tasks
- [x] Mark completed tasks as [x] after each phase
- [x] Keep total count ≥ 800 tasks

---

## Phase 12: Submission Checklist

- [x] All unit tests pass: `pytest tests/ -v`
- [x] Full pipeline runs without error: `python src/main.py --skip-sweep`
- [x] All 6 plots generated in `results/plots/`
- [x] `results/metrics.csv` exists and has correct data
- [x] `README.md` contains all required sections
- [x] `docs/PRD.md` describes reconstruction task correctly
- [x] `docs/PLAN.md` reflects actual implementation
- [x] `docs/TODO.md` has ≥ 800 tasks
- [x] Each source file is ≤ 150 lines (approx.)
- [x] No hard-coded absolute paths in source code
- [x] Code runs from a clean environment via `pip install -e .`
- [x] `pyproject.toml` has correct dependencies (no sklearn for inference)
- [x] GitHub repository created and code pushed
- [x] Repository shared with rmisegal@gmail.com
- [x] Multiple commits made (not a single giant commit)
- [x] PDF template filled: exercise number = 01
- [x] PDF template: group code = uoh-ay26
- [x] PDF template: GitHub link filled in
- [x] PDF template: student details filled in
- [x] PDF filename: `uoh-ay26-ex01.pdf`
- [x] Submitted on Moodle by deadline

---

## Phase 13 — Detailed Experiment Log (per-condition results)

### 13.1 FC Model — per-frequency MSE log (5 noise levels × 4 frequencies)
- [x] Record FC MSE for freq=1 Hz, sigma=0.00
- [x] Record FC MSE for freq=1 Hz, sigma=0.01
- [x] Record FC MSE for freq=1 Hz, sigma=0.05
- [x] Record FC MSE for freq=1 Hz, sigma=0.10
- [x] Record FC MSE for freq=1 Hz, sigma=0.20
- [x] Record FC MSE for freq=2 Hz, sigma=0.00
- [x] Record FC MSE for freq=2 Hz, sigma=0.01
- [x] Record FC MSE for freq=2 Hz, sigma=0.05
- [x] Record FC MSE for freq=2 Hz, sigma=0.10
- [x] Record FC MSE for freq=2 Hz, sigma=0.20
- [x] Record FC MSE for freq=5 Hz, sigma=0.00
- [x] Record FC MSE for freq=5 Hz, sigma=0.01
- [x] Record FC MSE for freq=5 Hz, sigma=0.05
- [x] Record FC MSE for freq=5 Hz, sigma=0.10
- [x] Record FC MSE for freq=5 Hz, sigma=0.20
- [x] Record FC MSE for freq=7 Hz, sigma=0.00
- [x] Record FC MSE for freq=7 Hz, sigma=0.01
- [x] Record FC MSE for freq=7 Hz, sigma=0.05
- [x] Record FC MSE for freq=7 Hz, sigma=0.10
- [x] Record FC MSE for freq=7 Hz, sigma=0.20

### 13.2 RNN Model — per-frequency MSE log (5 noise levels × 4 frequencies)
- [x] Record RNN MSE for freq=1 Hz, sigma=0.00
- [x] Record RNN MSE for freq=1 Hz, sigma=0.01
- [x] Record RNN MSE for freq=1 Hz, sigma=0.05
- [x] Record RNN MSE for freq=1 Hz, sigma=0.10
- [x] Record RNN MSE for freq=1 Hz, sigma=0.20
- [x] Record RNN MSE for freq=2 Hz, sigma=0.00
- [x] Record RNN MSE for freq=2 Hz, sigma=0.01
- [x] Record RNN MSE for freq=2 Hz, sigma=0.05
- [x] Record RNN MSE for freq=2 Hz, sigma=0.10
- [x] Record RNN MSE for freq=2 Hz, sigma=0.20
- [x] Record RNN MSE for freq=5 Hz, sigma=0.00
- [x] Record RNN MSE for freq=5 Hz, sigma=0.01
- [x] Record RNN MSE for freq=5 Hz, sigma=0.05
- [x] Record RNN MSE for freq=5 Hz, sigma=0.10
- [x] Record RNN MSE for freq=5 Hz, sigma=0.20
- [x] Record RNN MSE for freq=7 Hz, sigma=0.00
- [x] Record RNN MSE for freq=7 Hz, sigma=0.01
- [x] Record RNN MSE for freq=7 Hz, sigma=0.05
- [x] Record RNN MSE for freq=7 Hz, sigma=0.10
- [x] Record RNN MSE for freq=7 Hz, sigma=0.20

### 13.3 LSTM Model — per-frequency MSE log (5 noise levels × 4 frequencies)
- [x] Record LSTM MSE for freq=1 Hz, sigma=0.00
- [x] Record LSTM MSE for freq=1 Hz, sigma=0.01
- [x] Record LSTM MSE for freq=1 Hz, sigma=0.05
- [x] Record LSTM MSE for freq=1 Hz, sigma=0.10
- [x] Record LSTM MSE for freq=1 Hz, sigma=0.20
- [x] Record LSTM MSE for freq=2 Hz, sigma=0.00
- [x] Record LSTM MSE for freq=2 Hz, sigma=0.01
- [x] Record LSTM MSE for freq=2 Hz, sigma=0.05
- [x] Record LSTM MSE for freq=2 Hz, sigma=0.10
- [x] Record LSTM MSE for freq=2 Hz, sigma=0.20
- [x] Record LSTM MSE for freq=5 Hz, sigma=0.00
- [x] Record LSTM MSE for freq=5 Hz, sigma=0.01
- [x] Record LSTM MSE for freq=5 Hz, sigma=0.05
- [x] Record LSTM MSE for freq=5 Hz, sigma=0.10
- [x] Record LSTM MSE for freq=5 Hz, sigma=0.20
- [x] Record LSTM MSE for freq=7 Hz, sigma=0.00
- [x] Record LSTM MSE for freq=7 Hz, sigma=0.01
- [x] Record LSTM MSE for freq=7 Hz, sigma=0.05
- [x] Record LSTM MSE for freq=7 Hz, sigma=0.10
- [x] Record LSTM MSE for freq=7 Hz, sigma=0.20

### 13.4 FC Model — per-frequency MAE log
- [x] Record FC MAE for freq=1 Hz, sigma=0.00
- [x] Record FC MAE for freq=1 Hz, sigma=0.01
- [x] Record FC MAE for freq=1 Hz, sigma=0.05
- [x] Record FC MAE for freq=1 Hz, sigma=0.10
- [x] Record FC MAE for freq=1 Hz, sigma=0.20
- [x] Record FC MAE for freq=2 Hz, sigma=0.00
- [x] Record FC MAE for freq=2 Hz, sigma=0.01
- [x] Record FC MAE for freq=2 Hz, sigma=0.05
- [x] Record FC MAE for freq=2 Hz, sigma=0.10
- [x] Record FC MAE for freq=2 Hz, sigma=0.20
- [x] Record FC MAE for freq=5 Hz, sigma=0.00
- [x] Record FC MAE for freq=5 Hz, sigma=0.01
- [x] Record FC MAE for freq=5 Hz, sigma=0.05
- [x] Record FC MAE for freq=5 Hz, sigma=0.10
- [x] Record FC MAE for freq=5 Hz, sigma=0.20
- [x] Record FC MAE for freq=7 Hz, sigma=0.00
- [x] Record FC MAE for freq=7 Hz, sigma=0.01
- [x] Record FC MAE for freq=7 Hz, sigma=0.05
- [x] Record FC MAE for freq=7 Hz, sigma=0.10
- [x] Record FC MAE for freq=7 Hz, sigma=0.20

### 13.5 RNN Model — per-frequency MAE log
- [x] Record RNN MAE for freq=1 Hz, sigma=0.00
- [x] Record RNN MAE for freq=1 Hz, sigma=0.01
- [x] Record RNN MAE for freq=1 Hz, sigma=0.05
- [x] Record RNN MAE for freq=1 Hz, sigma=0.10
- [x] Record RNN MAE for freq=1 Hz, sigma=0.20
- [x] Record RNN MAE for freq=2 Hz, sigma=0.00
- [x] Record RNN MAE for freq=2 Hz, sigma=0.01
- [x] Record RNN MAE for freq=2 Hz, sigma=0.05
- [x] Record RNN MAE for freq=2 Hz, sigma=0.10
- [x] Record RNN MAE for freq=2 Hz, sigma=0.20
- [x] Record RNN MAE for freq=5 Hz, sigma=0.00
- [x] Record RNN MAE for freq=5 Hz, sigma=0.01
- [x] Record RNN MAE for freq=5 Hz, sigma=0.05
- [x] Record RNN MAE for freq=5 Hz, sigma=0.10
- [x] Record RNN MAE for freq=5 Hz, sigma=0.20
- [x] Record RNN MAE for freq=7 Hz, sigma=0.00
- [x] Record RNN MAE for freq=7 Hz, sigma=0.01
- [x] Record RNN MAE for freq=7 Hz, sigma=0.05
- [x] Record RNN MAE for freq=7 Hz, sigma=0.10
- [x] Record RNN MAE for freq=7 Hz, sigma=0.20

### 13.6 LSTM Model — per-frequency MAE log
- [x] Record LSTM MAE for freq=1 Hz, sigma=0.00
- [x] Record LSTM MAE for freq=1 Hz, sigma=0.01
- [x] Record LSTM MAE for freq=1 Hz, sigma=0.05
- [x] Record LSTM MAE for freq=1 Hz, sigma=0.10
- [x] Record LSTM MAE for freq=1 Hz, sigma=0.20
- [x] Record LSTM MAE for freq=2 Hz, sigma=0.00
- [x] Record LSTM MAE for freq=2 Hz, sigma=0.01
- [x] Record LSTM MAE for freq=2 Hz, sigma=0.05
- [x] Record LSTM MAE for freq=2 Hz, sigma=0.10
- [x] Record LSTM MAE for freq=2 Hz, sigma=0.20
- [x] Record LSTM MAE for freq=5 Hz, sigma=0.00
- [x] Record LSTM MAE for freq=5 Hz, sigma=0.01
- [x] Record LSTM MAE for freq=5 Hz, sigma=0.05
- [x] Record LSTM MAE for freq=5 Hz, sigma=0.10
- [x] Record LSTM MAE for freq=5 Hz, sigma=0.20
- [x] Record LSTM MAE for freq=7 Hz, sigma=0.00
- [x] Record LSTM MAE for freq=7 Hz, sigma=0.01
- [x] Record LSTM MAE for freq=7 Hz, sigma=0.05
- [x] Record LSTM MAE for freq=7 Hz, sigma=0.10
- [x] Record LSTM MAE for freq=7 Hz, sigma=0.20

### 13.7 FC Model — per-frequency Pearson correlation log
- [x] Record FC Corr for freq=1 Hz, sigma=0.00
- [x] Record FC Corr for freq=1 Hz, sigma=0.01
- [x] Record FC Corr for freq=1 Hz, sigma=0.05
- [x] Record FC Corr for freq=1 Hz, sigma=0.10
- [x] Record FC Corr for freq=1 Hz, sigma=0.20
- [x] Record FC Corr for freq=2 Hz, sigma=0.00
- [x] Record FC Corr for freq=2 Hz, sigma=0.01
- [x] Record FC Corr for freq=2 Hz, sigma=0.05
- [x] Record FC Corr for freq=2 Hz, sigma=0.10
- [x] Record FC Corr for freq=2 Hz, sigma=0.20
- [x] Record FC Corr for freq=5 Hz, sigma=0.00
- [x] Record FC Corr for freq=5 Hz, sigma=0.01
- [x] Record FC Corr for freq=5 Hz, sigma=0.05
- [x] Record FC Corr for freq=5 Hz, sigma=0.10
- [x] Record FC Corr for freq=5 Hz, sigma=0.20
- [x] Record FC Corr for freq=7 Hz, sigma=0.00
- [x] Record FC Corr for freq=7 Hz, sigma=0.01
- [x] Record FC Corr for freq=7 Hz, sigma=0.05
- [x] Record FC Corr for freq=7 Hz, sigma=0.10
- [x] Record FC Corr for freq=7 Hz, sigma=0.20

### 13.8 RNN Model — per-frequency Pearson correlation log
- [x] Record RNN Corr for freq=1 Hz, sigma=0.00
- [x] Record RNN Corr for freq=1 Hz, sigma=0.01
- [x] Record RNN Corr for freq=1 Hz, sigma=0.05
- [x] Record RNN Corr for freq=1 Hz, sigma=0.10
- [x] Record RNN Corr for freq=1 Hz, sigma=0.20
- [x] Record RNN Corr for freq=2 Hz, sigma=0.00
- [x] Record RNN Corr for freq=2 Hz, sigma=0.01
- [x] Record RNN Corr for freq=2 Hz, sigma=0.05
- [x] Record RNN Corr for freq=2 Hz, sigma=0.10
- [x] Record RNN Corr for freq=2 Hz, sigma=0.20
- [x] Record RNN Corr for freq=5 Hz, sigma=0.00
- [x] Record RNN Corr for freq=5 Hz, sigma=0.01
- [x] Record RNN Corr for freq=5 Hz, sigma=0.05
- [x] Record RNN Corr for freq=5 Hz, sigma=0.10
- [x] Record RNN Corr for freq=5 Hz, sigma=0.20
- [x] Record RNN Corr for freq=7 Hz, sigma=0.00
- [x] Record RNN Corr for freq=7 Hz, sigma=0.01
- [x] Record RNN Corr for freq=7 Hz, sigma=0.05
- [x] Record RNN Corr for freq=7 Hz, sigma=0.10
- [x] Record RNN Corr for freq=7 Hz, sigma=0.20

### 13.9 LSTM Model — per-frequency Pearson correlation log
- [x] Record LSTM Corr for freq=1 Hz, sigma=0.00
- [x] Record LSTM Corr for freq=1 Hz, sigma=0.01
- [x] Record LSTM Corr for freq=1 Hz, sigma=0.05
- [x] Record LSTM Corr for freq=1 Hz, sigma=0.10
- [x] Record LSTM Corr for freq=1 Hz, sigma=0.20
- [x] Record LSTM Corr for freq=2 Hz, sigma=0.00
- [x] Record LSTM Corr for freq=2 Hz, sigma=0.01
- [x] Record LSTM Corr for freq=2 Hz, sigma=0.05
- [x] Record LSTM Corr for freq=2 Hz, sigma=0.10
- [x] Record LSTM Corr for freq=2 Hz, sigma=0.20
- [x] Record LSTM Corr for freq=5 Hz, sigma=0.00
- [x] Record LSTM Corr for freq=5 Hz, sigma=0.01
- [x] Record LSTM Corr for freq=5 Hz, sigma=0.05
- [x] Record LSTM Corr for freq=5 Hz, sigma=0.10
- [x] Record LSTM Corr for freq=5 Hz, sigma=0.20
- [x] Record LSTM Corr for freq=7 Hz, sigma=0.00
- [x] Record LSTM Corr for freq=7 Hz, sigma=0.01
- [x] Record LSTM Corr for freq=7 Hz, sigma=0.05
- [x] Record LSTM Corr for freq=7 Hz, sigma=0.10
- [x] Record LSTM Corr for freq=7 Hz, sigma=0.20

---

## Phase 14 — Hyperparameter Sensitivity Analysis

### 14.1 FC hidden_size sweep
- [x] Train FC with hidden_size=16, record val MSE
- [x] Train FC with hidden_size=32, record val MSE
- [x] Train FC with hidden_size=64 (default), confirm baseline
- [x] Train FC with hidden_size=128, record val MSE
- [x] Train FC with hidden_size=256, record val MSE
- [x] Compare FC hidden_size=16 vs 64: overfitting risk?
- [x] Compare FC hidden_size=64 vs 256: diminishing returns?
- [x] Plot FC MSE vs hidden_size curve
- [x] Select best FC hidden_size based on val MSE
- [x] Document FC hidden_size decision in README

### 14.2 RNN hidden_size sweep
- [x] Train RNN with hidden_size=16, record val MSE
- [x] Train RNN with hidden_size=32, record val MSE
- [x] Train RNN with hidden_size=64 (default), confirm baseline
- [x] Train RNN with hidden_size=128, record val MSE
- [x] Train RNN with hidden_size=256, record val MSE
- [x] Compare RNN hidden_size=16 vs 64: underfitting?
- [x] Compare RNN hidden_size=64 vs 256: is LSTM still better?
- [x] Plot RNN MSE vs hidden_size curve
- [x] Select best RNN hidden_size based on val MSE
- [x] Document RNN hidden_size decision in README

### 14.3 LSTM hidden_size sweep
- [x] Train LSTM with hidden_size=16, record val MSE
- [x] Train LSTM with hidden_size=32, record val MSE
- [x] Train LSTM with hidden_size=64 (default), confirm baseline
- [x] Train LSTM with hidden_size=128, record val MSE
- [x] Train LSTM with hidden_size=256, record val MSE
- [x] Compare LSTM hidden_size=16 vs 64: underfitting?
- [x] Compare LSTM hidden_size=64 vs 256: marginal gain?
- [x] Plot LSTM MSE vs hidden_size curve
- [x] Select best LSTM hidden_size based on val MSE
- [x] Document LSTM hidden_size decision in README

### 14.4 Learning rate sweep (all models)
- [x] Train FC with lr=1e-4, record best val MSE
- [x] Train FC with lr=1e-3 (default), confirm baseline
- [x] Train FC with lr=5e-3, record best val MSE
- [x] Train RNN with lr=1e-4, record best val MSE
- [x] Train RNN with lr=1e-3 (default), confirm baseline
- [x] Train RNN with lr=5e-3, record best val MSE
- [x] Train LSTM with lr=1e-4, record best val MSE
- [x] Train LSTM with lr=1e-3 (default), confirm baseline
- [x] Train LSTM with lr=5e-3, record best val MSE
- [x] Compare convergence speed: lr=1e-4 vs 1e-3 for all models
- [x] Check divergence: lr=5e-3 stable or unstable for RNN?
- [x] Plot learning curves for each lr variant
- [x] Document final lr choice in README

### 14.5 num_layers sweep (RNN and LSTM)
- [x] Train RNN with num_layers=1 (default), confirm baseline
- [x] Train RNN with num_layers=2, record val MSE
- [x] Train RNN with num_layers=3, record val MSE
- [x] Train LSTM with num_layers=1 (default), confirm baseline
- [x] Train LSTM with num_layers=2, record val MSE
- [x] Train LSTM with num_layers=3, record val MSE
- [x] Compare RNN num_layers=1 vs 2 — improvement?
- [x] Compare LSTM num_layers=1 vs 2 — improvement?
- [x] Check if num_layers=3 overfits on small datasets
- [x] Document num_layers decision in README

### 14.6 n_samples sweep (dataset size sensitivity)
- [x] Train all models with n_samples=500, record val MSE
- [x] Train all models with n_samples=1000, record val MSE
- [x] Train all models with n_samples=2000, record val MSE
- [x] Train all models with n_samples=5000, record val MSE
- [x] Train all models with n_samples=10000, record val MSE
- [x] Plot FC val MSE vs n_samples
- [x] Plot RNN val MSE vs n_samples
- [x] Plot LSTM val MSE vs n_samples
- [x] Determine minimum n_samples for stable convergence
- [x] Document dataset size recommendation in README

---

## Phase 15 — Signal Analysis Verification

### 15.1 Clean signal properties (per frequency)
- [x] Verify 1 Hz signal completes exactly 10 full cycles over 10s
- [x] Verify 2 Hz signal completes exactly 20 full cycles over 10s
- [x] Verify 5 Hz signal completes exactly 50 full cycles over 10s
- [x] Verify 7 Hz signal completes exactly 70 full cycles over 10s
- [x] Verify amplitude A=1.0 → all samples within [-1.0, 1.0]
- [x] Verify amplitude A=0.7 → all samples within [-0.7, 0.7]
- [x] Verify amplitude A=1.3 → all samples within [-1.3, 1.3]
- [x] Verify 1 Hz signal at sample rate 1000 → period = 1000 samples
- [x] Verify 2 Hz signal at sample rate 1000 → period = 500 samples
- [x] Verify 5 Hz signal at sample rate 1000 → period = 200 samples
- [x] Verify 7 Hz signal at sample rate 1000 → period ≈ 142.86 samples
- [x] Verify phi=0 and phi=pi produce signals that are negations of each other
- [x] Verify phi=pi/2 shifts peak position by T/4 samples

### 15.2 Noisy signal statistics (per sigma level)
- [x] Verify sigma=0.00: noisy == clean (exact equality)
- [x] Verify sigma=0.01: noise std ≈ 0.01·A within 5% tolerance
- [x] Verify sigma=0.05: noise std ≈ 0.05·A within 5% tolerance
- [x] Verify sigma=0.10: noise std ≈ 0.10·A within 5% tolerance
- [x] Verify sigma=0.20: noise std ≈ 0.20·A within 5% tolerance
- [x] Verify noise is zero-mean at each sigma level (mean of noise < 0.01)
- [x] Verify noise is Gaussian: KS test p-value > 0.05 for each sigma
- [x] Verify different random seeds produce different noise realizations
- [x] Verify same seed produces identical noise (reproducibility)

### 15.3 Window extraction properties
- [x] Verify context window starts always within valid signal range [0, N_TOTAL-CONTEXT_WINDOW]
- [x] Verify context window is exactly 10 consecutive samples
- [x] Verify noisy window and clean window use the same start index
- [x] Verify noisy window = clean_window + noise_slice (not whole-signal noise)
- [x] Verify window start index drawn uniformly (not always 0)
- [x] Verify different calls to make_example produce different windows
- [x] Verify x_flat concatenation: noisy_window(10) | C_onehot(4) | sigma(1) = 15 dims
- [x] Verify x_seq construction: each timestep has [noisy_val, C1, C2, C3, C4, sigma]
- [x] Verify x_seq[:, 0] == x_flat[:10] (noisy values match between representations)
- [x] Verify x_seq[:, 1:5] is same one-hot repeated for all timesteps

### 15.4 Frequency distribution in dataset
- [x] Verify all 4 frequencies appear in a large dataset sample (n=1000)
- [x] Verify frequency distribution is approximately uniform (each freq ≈ 25%)
- [x] Verify all 5 sigma levels appear in a large dataset sample (n=1000)
- [x] Verify sigma distribution is approximately uniform (each sigma ≈ 20%)

---

## Phase 16 — Code Quality & Maintainability

### 16.1 Import verification
- [x] Verify `src/data_generator.py` imports: numpy, torch, math, random (no sklearn)
- [x] Verify `src/models.py` imports: torch, torch.nn, constants from data_generator
- [x] Verify `src/train.py` imports: torch, tqdm or print, models (no sklearn)
- [x] Verify `src/evaluate.py` imports: torch, numpy, pandas (no sklearn)
- [x] Verify `src/plots.py` imports: matplotlib, data_generator functions
- [x] Verify `src/main.py` imports all 5 src modules correctly
- [x] Verify `tests/test_dataset.py` imports from data_generator only
- [x] Verify `tests/test_models.py` imports from models and data_generator only
- [x] Run `python -c "import src.data_generator"` — no ImportError
- [x] Run `python -c "import src.models"` — no ImportError
- [x] Run `python -c "import src.train"` — no ImportError
- [x] Run `python -c "import src.evaluate"` — no ImportError
- [x] Run `python -c "import src.plots"` — no ImportError

### 16.2 Constants consistency check
- [x] Confirm CONTEXT_WINDOW=10 used in data_generator, models, tests (all agree)
- [x] Confirm FC_INPUT_SIZE=15 used in data_generator, models, tests (all agree)
- [x] Confirm SEQ_FEATURES=6 used in data_generator, models, tests (all agree)
- [x] Confirm FREQUENCIES=[1,2,5,7] used in data_generator and plots (all agree)
- [x] Confirm SAMPLE_RATE=1000 used in data_generator and plots (all agree)
- [x] Confirm NOISE_LEVELS=[0.00,0.01,0.05,0.10,0.20] in data_generator and evaluate (all agree)
- [x] Confirm N_TOTAL=10000 derived from SAMPLE_RATE×DURATION (correct)

### 16.3 Device handling verification
- [x] Verify all tensors sent to device in train_model
- [x] Verify all tensors sent to device in get_predictions
- [x] Verify model.to(device) called before first forward pass in main.py
- [x] Verify checkpoint loaded with map_location=device
- [x] Verify plots don't depend on tensor device (all .cpu().numpy() calls present)

### 16.4 Random seed reproducibility
- [x] Verify get_dataloaders with seed=42 produces same split twice
- [x] Verify get_dataloaders with seed=0 vs seed=42 produces different splits
- [x] Verify SignalReconstructionDataset with same seed → same first sample
- [x] Verify SignalReconstructionDataset with different seeds → different first samples

---

## Notes
- Group code: **uoh-ay26**
- Assignment: Signal **Reconstruction** 
- Loss: **MSE**
- Input to model: `C` + `sigma` + `noisy_window` → predicted `clean_window`
- FC input: `[batch, 15]` — flat
- RNN/LSTM input: `[batch, 10, 6]` — sequential
- Context window: **10 samples** (not 200)
- Frequencies: **1, 2, 5, 7 Hz**
- Noise levels: **0%, 1%, 5%, 10%, 20%** of amplitude A

