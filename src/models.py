"""FC, bidirectional RNN, and bidirectional LSTM reconstruction models."""

import torch
import torch.nn as nn

from .data_generator import CONTEXT_WINDOW, FC_INPUT_SIZE, SEQ_FEATURES


class FCNet(nn.Module):
    """Flat baseline: [batch, 105] -> [batch, 100]."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FC_INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, CONTEXT_WINDOW),
        )

    def forward(self, x_flat: torch.Tensor, x_seq: torch.Tensor = None) -> torch.Tensor:
        return self.net(x_flat)


def _init_recurrent_weights(module: nn.Module) -> None:
    """Orthogonal recurrent weights and zero bias improve early stability."""
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


class RNNNet(nn.Module):
    """Bidirectional many-to-many RNN: [batch, 100, 6] -> [batch, 100]."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            SEQ_FEATURES,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        _init_recurrent_weights(self)

    def forward(self, x_flat: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_seq)
        return self.fc(self.layer_norm(out)).squeeze(-1)


class LSTMNet(nn.Module):
    """Bidirectional many-to-many LSTM: [batch, 100, 6] -> [batch, 100]."""

    def __init__(self, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            SEQ_FEATURES,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        _init_recurrent_weights(self)

    def forward(self, x_flat: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        return self.fc(self.layer_norm(out)).squeeze(-1)
