"""
tests/test_models.py
--------------------
Unit tests for models.py (signal reconstruction: MSE task).

Coverage
--------
- Output shapes for FC, RNN, LSTM
- FC uses x_flat; RNN/LSTM use x_seq
- Gradient flow through all parameters
- Parameter count sanity (LSTM > RNN)
- Determinism with fixed seed
- MSE loss can be computed on model output
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from models import (
    CONTEXT_WINDOW,
    FC_INPUT_SIZE,
    SEQ_FEATURES,
    FCNet,
    LSTMNet,
    RNNNet,
)

BATCH = 8


@pytest.fixture
def x_flat():
    return torch.randn(BATCH, FC_INPUT_SIZE)


@pytest.fixture
def x_seq():
    return torch.randn(BATCH, CONTEXT_WINDOW, SEQ_FEATURES)


# ── Output shapes ─────────────────────────────────────────────────────────────

def test_fc_output_shape(x_flat, x_seq):
    out = FCNet()(x_flat, x_seq)
    assert out.shape == (BATCH, CONTEXT_WINDOW), \
        f"FC output: expected ({BATCH},{CONTEXT_WINDOW}), got {out.shape}"


def test_rnn_output_shape(x_flat, x_seq):
    out = RNNNet()(x_flat, x_seq)
    assert out.shape == (BATCH, CONTEXT_WINDOW), \
        f"RNN output: expected ({BATCH},{CONTEXT_WINDOW}), got {out.shape}"


def test_lstm_output_shape(x_flat, x_seq):
    out = LSTMNet()(x_flat, x_seq)
    assert out.shape == (BATCH, CONTEXT_WINDOW), \
        f"LSTM output: expected ({BATCH},{CONTEXT_WINDOW}), got {out.shape}"


# ── FC ignores x_seq (can pass None) ─────────────────────────────────────────

def test_fc_accepts_none_x_seq(x_flat):
    out = FCNet()(x_flat, None)
    assert out.shape == (BATCH, CONTEXT_WINDOW)


# ── MSE loss can be computed ──────────────────────────────────────────────────

@pytest.mark.parametrize("ModelClass,use_seq", [
    (FCNet,   False),
    (RNNNet,  True),
    (LSTMNet, True),
])
def test_mse_loss_computable(ModelClass, use_seq, x_flat, x_seq):
    model  = ModelClass()
    target = torch.zeros(BATCH, CONTEXT_WINDOW)
    pred   = model(x_flat, x_seq if use_seq else None)
    loss   = nn.MSELoss()(pred, target)
    assert loss.item() >= 0.0


# ── Gradient flow ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ModelClass,use_seq", [
    (FCNet,   False),
    (RNNNet,  True),
    (LSTMNet, True),
])
def test_gradients_flow(ModelClass, use_seq, x_flat, x_seq):
    model  = ModelClass()
    target = torch.zeros(BATCH, CONTEXT_WINDOW)
    pred   = model(x_flat, x_seq if use_seq else None)
    nn.MSELoss()(pred, target).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


# ── Parameter counts ──────────────────────────────────────────────────────────

def test_all_models_have_parameters():
    for cls in (FCNet, RNNNet, LSTMNet):
        n = sum(p.numel() for p in cls().parameters() if p.requires_grad)
        assert n > 0, f"{cls.__name__} has no trainable parameters"


def test_lstm_has_more_params_than_rnn():
    """LSTM has 4 gates vs RNN's 1 → roughly 4× more parameters."""
    h    = 64
    n_rnn  = sum(p.numel() for p in RNNNet (hidden_size=h).parameters())
    n_lstm = sum(p.numel() for p in LSTMNet(hidden_size=h).parameters())
    assert n_lstm > n_rnn


# ── Determinism ───────────────────────────────────────────────────────────────

def test_fc_deterministic(x_flat, x_seq):
    # eval() disables Dropout and uses running BN stats → deterministic
    torch.manual_seed(0)
    m = FCNet(); m.eval()
    out1 = m(x_flat, None)
    torch.manual_seed(0)
    m2 = FCNet(); m2.eval()
    out2 = m2(x_flat, None)
    assert torch.allclose(out1, out2)


def test_rnn_deterministic(x_flat, x_seq):
    # eval() disables inter-layer dropout → deterministic
    torch.manual_seed(0)
    m = RNNNet(); m.eval()
    out1 = m(x_flat, x_seq)
    torch.manual_seed(0)
    m2 = RNNNet(); m2.eval()
    out2 = m2(x_flat, x_seq)
    assert torch.allclose(out1, out2)


def test_lstm_deterministic(x_flat, x_seq):
    # eval() disables inter-layer dropout → deterministic
    torch.manual_seed(0)
    m = LSTMNet(); m.eval()
    out1 = m(x_flat, x_seq)
    torch.manual_seed(0)
    m2 = LSTMNet(); m2.eval()
    out2 = m2(x_flat, x_seq)
    assert torch.allclose(out1, out2)


# ── hidden_size parameter honoured ───────────────────────────────────────────

def test_rnn_hidden_size_64(x_flat, x_seq):
    """Non-default hidden_size still produces correct output shape."""
    model = RNNNet(hidden_size=64, num_layers=1)
    out   = model(x_flat, x_seq)
    assert out.shape == (BATCH, CONTEXT_WINDOW)


def test_lstm_hidden_size_64(x_flat, x_seq):
    """Non-default hidden_size still produces correct output shape."""
    model = LSTMNet(hidden_size=64, num_layers=1)
    out   = model(x_flat, x_seq)
    assert out.shape == (BATCH, CONTEXT_WINDOW)

