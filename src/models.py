"""
models.py
---------
Three architectures for signal RECONSTRUCTION (denoising, MSE loss).

  FCNet   — Fully Connected baseline
            input  [batch, 105]  = [mixed_window(100) | C(4) | sigma]
            output [batch, 100]  = predicted clean window

  RNNNet  — Vanilla RNN (Many-to-Many)
            input  [batch, 100, 6] per-step [mixed_val | C1,C2,C3,C4 | sigma]
            output [batch, 100]

  LSTMNet — LSTM (gated memory, Many-to-Many)
            input  [batch, 100, 6]
            output [batch, 100]

Architecture improvements (inspired by reference best-practices):
  - FCNet   : BatchNorm1d + Dropout(0.1) for training stability
  - RNNNet  : LayerNorm + orthogonal weight init + inter-layer dropout
  - LSTMNet : LayerNorm + orthogonal weight init + inter-layer dropout

Loss: MSELoss(prediction, clean_window)
"""

import torch
import torch.nn as nn
from data_generator import FC_INPUT_SIZE, SEQ_FEATURES, CONTEXT_WINDOW


# ── Fully Connected ───────────────────────────────────────────────────────────

class FCNet(nn.Module):
    """
    Fully Connected baseline (no temporal awareness).

    Treats the entire 50-sample window as a flat, unordered feature vector.
    One hidden layer, ReLU activation — intentionally simple so that
    sequential models (RNN, LSTM) can outperform it by exploiting time structure.

    Architecture:
        Linear(FC_INPUT_SIZE, h) → ReLU → Linear(h, CONTEXT_WINDOW)

    Strength : fast to train; the condition inputs (C, sigma) tell it almost
               everything about the signal type.
    Weakness : no temporal ordering — treats the 50 values as a bag;
               cannot track phase, curvature, or multi-cycle patterns.
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        h = hidden_size
        self.net = nn.Sequential(
            nn.Linear(FC_INPUT_SIZE, h),
            nn.ReLU(),
            nn.Linear(h, CONTEXT_WINDOW),
        )

    def forward(self, x_flat: torch.Tensor, x_seq: torch.Tensor = None) -> torch.Tensor:
        # x_flat: [batch, FC_INPUT_SIZE]
        return self.net(x_flat)   # [batch, CONTEXT_WINDOW]


# ── Shared weight initialisation ──────────────────────────────────────────────

def _init_recurrent_weights(module: nn.Module) -> None:
    """
    Orthogonal initialisation for all weight matrices, zero bias.

    Orthogonal matrices preserve gradient norms at initialisation,
    giving RNN/LSTM training a stable starting point (avoids early
    vanishing / exploding gradients before the gates learn to regulate).
    """
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


# ── Vanilla RNN ───────────────────────────────────────────────────────────────

class RNNNet(nn.Module):
    """
    Simple Recurrent Neural Network (Elman RNN), Many-to-Many.

    Processes the 10-sample noisy window step-by-step together with the
    frequency condition C and noise level sigma repeated at every time step.

    Hidden state: h_t = tanh(W_h·h_{t-1} + W_x·x_t + b)

    Architecture:
        bidirectional RNN(input=6, hidden=h, num_layers=2, dropout=0.1, batch_first=True)
        LayerNorm(2h)
        Linear(2h, 1) at every time step

    LayerNorm stabilises hidden states across the sequence.
    Orthogonal init (see _init_recurrent_weights) prevents early gradient issues.
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            SEQ_FEATURES, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc         = nn.Linear(hidden_size * 2, 1)
        _init_recurrent_weights(self)

    def forward(self, x_flat: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [batch, CONTEXT_WINDOW, SEQ_FEATURES]
        out, _ = self.rnn(x_seq)        # [batch, CONTEXT_WINDOW, hidden*2]
        out    = self.layer_norm(out)   # stable normalisation
        out    = self.fc(out)           # [batch, CONTEXT_WINDOW, 1]
        return out.squeeze(-1)          # [batch, CONTEXT_WINDOW]


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMNet(nn.Module):
    """
    Long Short-Term Memory Network, Many-to-Many.

    Adds a cell state C_t alongside h_t, controlled by forget / input /
    output gates — solves the vanishing-gradient problem of the plain RNN.

    Gates:
        f_t = σ(W_f·[h_{t-1}, x_t] + b_f)   ← what to erase from cell
        i_t = σ(W_i·[h_{t-1}, x_t] + b_i)   ← what to write to cell
        o_t = σ(W_o·[h_{t-1}, x_t] + b_o)   ← what to expose as output
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_C·[h_{t-1},x_t]+b_C)
        h_t = o_t ⊙ tanh(C_t)

    When f_t ≈ 1: ∂C_t/∂C_{t-1} ≈ 1 — gradients flow without vanishing.

    Architecture:
        bidirectional LSTM(input=6, hidden=h, num_layers=2, dropout=0.1, batch_first=True)
        LayerNorm(2h)
        Linear(2h, 1) at every time step
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            SEQ_FEATURES, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc         = nn.Linear(hidden_size * 2, 1)
        _init_recurrent_weights(self)

    def forward(self, x_flat: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [batch, CONTEXT_WINDOW, SEQ_FEATURES]
        out, _ = self.lstm(x_seq)       # [batch, CONTEXT_WINDOW, hidden*2]
        out    = self.layer_norm(out)   # stable normalisation
        out    = self.fc(out)           # [batch, CONTEXT_WINDOW, 1]
        return out.squeeze(-1)          # [batch, CONTEXT_WINDOW]
