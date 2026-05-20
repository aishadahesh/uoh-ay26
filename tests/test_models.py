"""Unit tests for reconstruction models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from src.models import CONTEXT_WINDOW, FC_INPUT_SIZE, SEQ_FEATURES, FCNet, LSTMNet, RNNNet

BATCH = 8


@pytest.fixture
def x_flat():
    return torch.randn(BATCH, FC_INPUT_SIZE)


@pytest.fixture
def x_seq():
    return torch.randn(BATCH, CONTEXT_WINDOW, SEQ_FEATURES)


def model_cases():
    return [(FCNet, False), (RNNNet, True), (LSTMNet, True)]


def test_fc_output_shape(x_flat, x_seq):
    assert FCNet()(x_flat, x_seq).shape == (BATCH, CONTEXT_WINDOW)


def test_rnn_output_shape(x_flat, x_seq):
    assert RNNNet()(x_flat, x_seq).shape == (BATCH, CONTEXT_WINDOW)


def test_lstm_output_shape(x_flat, x_seq):
    assert LSTMNet()(x_flat, x_seq).shape == (BATCH, CONTEXT_WINDOW)


def test_fc_accepts_none_x_seq(x_flat):
    assert FCNet()(x_flat, None).shape == (BATCH, CONTEXT_WINDOW)


@pytest.mark.parametrize("ModelClass,use_seq", model_cases())
def test_mse_loss_computable(ModelClass, use_seq, x_flat, x_seq):
    model = ModelClass()
    target = torch.zeros(BATCH, CONTEXT_WINDOW)
    pred = model(x_flat, x_seq if use_seq else None)
    assert nn.MSELoss()(pred, target).item() >= 0.0


@pytest.mark.parametrize("ModelClass,use_seq", model_cases())
def test_gradients_flow(ModelClass, use_seq, x_flat, x_seq):
    model = ModelClass()
    target = torch.zeros(BATCH, CONTEXT_WINDOW)
    pred = model(x_flat, x_seq if use_seq else None)
    nn.MSELoss()(pred, target).backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_all_models_have_parameters():
    for cls in (FCNet, RNNNet, LSTMNet):
        n_params = sum(p.numel() for p in cls().parameters() if p.requires_grad)
        assert n_params > 0


def test_lstm_has_more_params_than_rnn():
    """LSTM has four gates while RNN has one recurrent transform."""
    hidden = 64
    n_rnn = sum(p.numel() for p in RNNNet(hidden_size=hidden).parameters())
    n_lstm = sum(p.numel() for p in LSTMNet(hidden_size=hidden).parameters())
    assert n_lstm > n_rnn


@pytest.mark.parametrize("ModelClass,use_seq", model_cases())
def test_models_deterministic_in_eval(ModelClass, use_seq, x_flat, x_seq):
    torch.manual_seed(0)
    model_a = ModelClass()
    model_a.eval()
    out_a = model_a(x_flat, x_seq if use_seq else None)
    torch.manual_seed(0)
    model_b = ModelClass()
    model_b.eval()
    out_b = model_b(x_flat, x_seq if use_seq else None)
    assert torch.allclose(out_a, out_b)


def test_rnn_hidden_size_64(x_flat, x_seq):
    assert RNNNet(hidden_size=64, num_layers=1)(x_flat, x_seq).shape == (
        BATCH,
        CONTEXT_WINDOW,
    )


def test_lstm_hidden_size_64(x_flat, x_seq):
    assert LSTMNet(hidden_size=64, num_layers=1)(x_flat, x_seq).shape == (
        BATCH,
        CONTEXT_WINDOW,
    )
