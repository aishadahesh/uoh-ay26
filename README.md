# Signal Reconstruction &mdash; FC vs RNN vs LSTM

<div align="center">

```
  noisy signal  -->  [ FC / RNN / LSTM ]  -->  clean signal
```

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Signal%20Reconstruction-blueviolet)
![Loss](https://img.shields.io/badge/Loss-MSE-success)
![Tests](https://img.shields.io/badge/Tests-45%20passing-brightgreen)
![Group](https://img.shields.io/badge/Group-uoh--ay26-orange)

**Assignment 01 &mdash; AI Agents Orchestration &mdash; Group `uoh-ay26`**

</div>

---

## What This Project Does

> **Task:** Given a short noisy window of a sinusoidal signal plus its condition
> (frequency class + noise level), **reconstruct the underlying clean window**.
> This is a **denoising / regression** task &mdash; not classification.

```
Input  -->  [ C1 C2 C3 C4 | sigma | y~1 y~2 ... y~10 ]   (15 values, flat)
                   |
             Neural Network
                   |
Output -->  [ y^1  y^2  y^3  y^4  y^5  y^6  y^7  y^8  y^9  y^10 ]  (10 clean samples)

Loss  =  MSE( prediction, clean_window )
```

Three architectures compete on the same task:

| Model | Input shape | Architecture | Params |
|-------|-------------|--------------|--------|
| **FC** &mdash; Fully Connected | `[batch, 15]` flat | Linear×3 + BatchNorm + Dropout | ~20,000 |
| **RNN** &mdash; Vanilla Recurrent | `[batch, 10, 6]` sequence | 2-layer RNN + LayerNorm + ortho-init | ~51,000 |
| **LSTM** &mdash; Long Short-Term Memory | `[batch, 10, 6]` sequence | 2-layer LSTM + LayerNorm + ortho-init | ~202,000 |

---

## The Signal Model

Each training example is generated as:

```
x(t) = A * sin(2*pi * f * t + phi) + eta(t)
```

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| `f` | in {1, 2, 5, 7} Hz | Signal frequency |
| `A` | Uniform(0.7, 1.3) | Per-example amplitude jitter |
| `phi` | Uniform(0, 2*pi) | Per-example random phase |
| `eta(t)` | N(0, (sigma*A)^2) | Gaussian noise scaled by amplitude |
| `sigma` | in {0.00, 0.01, 0.05, 0.10, 0.20} | Noise level as fraction of A |

**Sampling:** 1000 Hz, total signal = 10,000 samples.
Each example draws a random **10-sample window** from the full signal.

---

## Signal Visualisation

> All four frequencies &mdash; clean vs noisy at sigma = 0.10:

![Signals](results/plots/signals.png)

> One 10-sample window &mdash; noisy input vs clean target:

![Window Example](results/plots/window_example.png)

---

## Architecture Details

### FC &mdash; Fully Connected Network

```
x_flat  [batch, 15]
    |
    +-- noisy_window  [10]  --+
    +-- one_hot C     [4]   --+-- flat concatenation
    +-- sigma         [1]  --+
          |
   Linear(15 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.1)
          |
   Linear(128 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.1)
          |
   Linear(128 -> 10)
          |
   y_hat  [batch, 10]
```

- **Parameters:** ~20,000
- **BatchNorm** normalises each hidden layer's activations &mdash; faster convergence, less sensitivity to learning rate.
- **Dropout(0.1)** prevents overfitting on repeated noise patterns.
- **Strength:** Fast, simple; the condition inputs (C, sigma) tell it almost everything.
- **Limitation:** No recurrence &mdash; treats the 10 noisy samples as an unordered bag of values.

---

### RNN &mdash; Vanilla Recurrent Network

```
x_seq  [batch, 10, 6]   -- one row per time step: [noisy_val, C1,C2,C3,C4, sigma]
    |
    v  (step by step, 2 stacked layers)
  h_t = tanh( W_h * h_{t-1}  +  W_x * x_t  +  b )    <- per layer
  Dropout(0.1) between layer 1 and layer 2
    |
  LayerNorm(128)       <- stabilises hidden states
    |
    +-- at every step: Linear(128 -> 1) --> scalar prediction
    |
  [y^1, y^2, ..., y^10]  stacked  -->  [batch, 10]
```

- **Parameters:** ~51,000
- **Orthogonal weight init** &mdash; preserves gradient norms at start; avoids early vanishing.
- **LayerNorm** normalises hidden states across the feature dimension each step.
- **Dropout(0.1)** applied between the two stacked RNN layers.
- **Key equation:** gradient proportional to product of Jacobians over 10 steps.

---

### LSTM &mdash; Long Short-Term Memory

```
x_seq  [batch, 10, 6]
    |
    v  (step by step, 2 stacked layers)
  f_t = sigmoid( W_f * [h_{t-1}, x_t] + b_f )   <- forget gate
  i_t = sigmoid( W_i * [h_{t-1}, x_t] + b_i )   <- input gate
  g_t = tanh(    W_g * [h_{t-1}, x_t] + b_g )
  C_t = f_t * C_{t-1}  +  i_t * g_t             <- cell state update
  o_t = sigmoid( W_o * [h_{t-1}, x_t] + b_o )   <- output gate
  h_t = o_t * tanh(C_t)
  Dropout(0.1) between layer 1 and layer 2
    |
  LayerNorm(128)       <- stabilises hidden states
    |
    +-- at every step: Linear(128 -> 1) --> scalar prediction
    |
  [y^1, y^2, ..., y^10]  stacked  -->  [batch, 10]
```

- **Parameters:** ~202,000
- **Advantage:** When `f_t ~ 1`, the cell-state gradient is also ~1 &mdash; no vanishing.
- **Orthogonal weight init** gives the gates a clean starting point.
- **LayerNorm** stabilises the output of each cell across all 10 steps.
- **Why better than RNN:** 4 gate matrices control the cell state explicitly &mdash; gradients can flow freely.

---

## Input Encoding

### FC path &mdash; flat vector `[15]`

```
Index:  0  1  2  3  4  5  6  7  8  9 | 10 11 12 13 | 14
        +---------------------------------+-----------+----+
Value:  y~1 y~2 ...             y~10  | C1 C2 C3 C4 | sigma
        +-- noisy window (10) --------+- one-hot(4)-+----+
```

### RNN / LSTM path &mdash; sequence `[10, 6]`

```
Timestep t:  [ y~_t  |  C1  C2  C3  C4  |  sigma ]   (6 features per step)
                 ^            ^                ^
            noisy val     freq class      noise level
            (changes)    (repeated)      (repeated)
```

---

## Training Setup

| Setting | Value |
|---------|-------|
| Loss | **MSELoss** `L = mean((y_hat - y_clean)^2)` |
| Optimizer | Adam (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | ReduceLROnPlateau (`patience=5`, `factor=0.5`) |
| Gradient clipping | `max_norm=1.0` |
| Epochs | 50 |
| Batch size | 64 |
| Train / Val / Test | 70% / 15% / 15% |
| Best checkpoint | Saved on minimum validation MSE |

---

## Results

### Training Loss Curves

> MSE decreases epoch-by-epoch for all three models:

![Training Loss](results/plots/training_loss.png)

### Prediction vs. Ground Truth

> A test-set example: noisy input, true clean signal, and model prediction:

![Prediction vs True](results/plots/prediction_vs_true.png)

### MSE per Frequency

> Which frequency is hardest to reconstruct?

![MSE per Frequency](results/plots/mse_per_frequency.png)

**Why 1 Hz is hardest:** A 10-sample window at 1000 Hz covers only **1% of one full period**.
The model sees an almost flat line &mdash; very little oscillation shape is visible.
At 7 Hz, 10 samples cover **70% of one period** &mdash; the shape is clearly recognisable.

| Frequency | Period (samples) | Coverage in 10-sample window |
|-----------|-----------------|------------------------------|
| 1 Hz | 1000 | 1% of a period &mdash; nearly flat |
| 2 Hz | 500 | 2% of a period &mdash; tiny slope |
| 5 Hz | 200 | 5% of a period &mdash; visible curve |
| 7 Hz | ~143 | ~7% of a period &mdash; clearest shape |

---

## Why MSE?

```
L = (1 / N*W) * sum over n,t of ( y_hat_{n,t} - y_{n,t} )^2

  N = batch size
  W = window size (10)
```

- Penalises large errors quadratically &mdash; outliers are heavily punished.
- Optimal predictor under Gaussian noise assumption (matches our noise model).
- Directly interpretable: MSE = 0 means perfect reconstruction.

**Baseline:** A model that always predicts **zero** achieves `MSE = E[A^2 / 2] ~= 0.54`
(average signal power at A ~ Uniform(0.7, 1.3)). Every trained model must beat this.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Train and evaluate all three models
python src/main.py

# Train a specific model
python src/main.py --model fc
python src/main.py --model rnn
python src/main.py --model lstm

# Custom settings
python src/main.py --model all --epochs 100 --n-samples 20000 --batch-size 128

# Skip the slow noise sweep
python src/main.py --skip-sweep
```

### Run tests

```bash
pytest tests/ -v
# Expected: 45 passed
```

---

## Project Structure

```
uoh-ay26/
|
+-- README.md                 <- You are here
+-- requirements.txt          <- pip install -r requirements.txt
+-- pyproject.toml            <- project metadata
|
+-- src/
|   +-- data_generator.py     <- signal generation, SignalReconstructionDataset
|   +-- models.py             <- FCNet, RNNNet, LSTMNet
|   +-- train.py              <- training loop (MSELoss, Adam, scheduler)
|   +-- evaluate.py           <- MSE/MAE/Corr metrics, noise sweep
|   +-- plots.py              <- all 6 visualisations
|   +-- main.py               <- CLI entry point
|
+-- tests/
|   +-- test_dataset.py       <- 28 dataset unit tests
|   +-- test_models.py        <- 17 model unit tests
|
+-- results/
|   +-- metrics.csv           <- all numeric results
|   +-- plots/
|       +-- signals.png
|       +-- window_example.png
|       +-- training_loss.png
|       +-- prediction_vs_true.png
|       +-- mse_per_frequency.png
|       +-- noise_vs_mse.png
|
+-- docs/
    +-- PRD.md                <- product requirements
    +-- PLAN.md               <- implementation plan
    +-- TODO.md               <- 900-task tracker
```

---

## Key Insights

| Observation | Explanation |
|-------------|-------------|
| At sigma=0, MSE near 0 | No noise &mdash; perfect reconstruction is trivial |
| 1 Hz is hardest | 10 samples = 1% of period &mdash; nearly flat window |
| 7 Hz is easiest | 10 samples = 70% of period &mdash; shape is visible |
| LSTM >= RNN | Gated cell state protects temporal structure |
| FC competitive | Explicit C and sigma inputs carry most of the information |
| MSE grows with sigma | Higher noise = harder to recover the true signal |

---

## Submission

| Field | Value |
|-------|-------|
| **Group code** | `uoh-ay26` |
| **Assignment** | 01 &mdash; Signal Reconstruction |
| **Submission file** | `uoh-ay26-ex01.pdf` |

---

<div align="center">

Built with PyTorch &middot; NumPy &middot; Matplotlib &middot; Pandas

</div>
