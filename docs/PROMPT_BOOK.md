# Prompt Book

A running log of the significant AI prompts used during development,
documenting context, outputs, and iterative improvements.

---

## Entry 1 — Mixed-Signal Separation Task Design

**Date:** 2026-05  
**Context:** Defining the core ML task for Assignment 01.

**Prompt:**
> Design a neural network task where the model separates one sinusoidal
> component from a noisy mixture of four sinusoids. The model receives
> the mixed window plus a one-hot selector and must output the clean
> target component.

**Output / Decision:**
- Signal model: `Mixed(t) = Σ_k [A_k·sin(2π·f_k·t + φ_k) + η_k(t)]`
- Frequencies: [1, 2, 5, 7] Hz; A_k ~ U(0.7, 1.3); φ_k ~ U(0, 2π)
- Per-component Gaussian noise η_k ~ N(0, σ·|A_k|)
- σ drawn from {0.0, 0.1, 0.3, 0.5, 1.0}
- Input: noisy_window(100) + one-hot C(4) + σ(1) = 105 features
- Target: clean_window(100) of selected component

**Iteration:** Switched from single-signal denoising to full mixed-signal
separation to make the selector C meaningful and the task harder.

---

## Entry 2 — Model Architecture Balancing

**Date:** 2026-05  
**Context:** Choosing hidden sizes so models are comparable in scale.

**Prompt:**
> Choose hidden_size for FC, RNN, and LSTM so that parameter counts are
> in the same order of magnitude (roughly 10K–40K params each).
> All use hidden=64 and bidirectional RNN/LSTM with 1 layer.

**Output:**
| Model | Params  | MSE    |
|-------|---------|--------|
| FC    | 13,284  | 0.2942 |
| LSTM  | 37,249  | 0.3131 |
| RNN   | 9,601   | 0.3609 |

**Insight:** FC outperforms recurrent models on this fixed-window task
because the flat feature vector provides direct access to all timesteps
simultaneously without the vanishing-gradient burden of sequences.

---

## Entry 3 — Enforcing 150-Line File Limit

**Date:** 2026-05  
**Context:** `data_generator.py` was 168 lines, exceeding the 150-line rule.

**Prompt:**
> Split `data_generator.py` into `signals.py` (pure-numpy primitives +
> constants) and `data_generator.py` (PyTorch Dataset/DataLoader).
> Re-export everything from `data_generator.py` for backward compatibility.

**Output:** `signals.py` (86 lines) + `data_generator.py` (106 lines).  
All 45 tests continued to pass after the split.

---

## Entry 4 — Package Structure & Relative Imports

**Date:** 2026-05  
**Context:** Phase 2 compliance — formal Python package with `__init__.py`.

**Prompt:**
> Add `src/__init__.py` as the SDK entry point, convert all intra-src
> absolute imports to relative imports, centralise `sys.path` setup in
> `tests/conftest.py`.

**Output:**
- `src/__init__.py` exposes FCNet, LSTMNet, RNNNet, get_dataloaders, constants.
- 17 import statements updated across 8 source files.
- 3 test files cleaned of boilerplate `sys.path.insert` blocks.

---

## Entry 5 — Test Coverage & Ruff Linting

**Date:** 2026-05  
**Context:** Phase 3/4 compliance — 85% coverage threshold, zero Ruff violations.

**Prompt:**
> Configure `pyproject.toml` with `--cov-fail-under=85` and `[tool.ruff]`
> lint rules (E, F, W, I). Add missing tests to reach the 85% threshold.

**Outcome:** Coverage brought to ≥ 85 %; Ruff configured with zero violations.
