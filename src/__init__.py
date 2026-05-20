"""uoh-ay26 – Signal component separation: FC vs RNN vs LSTM.

SDK entry point.  Import models, dataset utilities, and signal primitives
directly from this package instead of reaching into submodules.

Example
-------
>>> from src import FCNet, get_dataloaders, NOISE_LEVELS
"""

from .data_generator import SignalReconstructionDataset, get_dataloaders
from .models import FCNet, LSTMNet, RNNNet
from .signals import (
    CONTEXT_WINDOW,
    FC_INPUT_SIZE,
    FREQUENCIES,
    N_FREQS,
    N_TOTAL,
    NOISE_LEVELS,
    SAMPLE_RATE,
    SEQ_FEATURES,
    add_gaussian_noise,
    generate_clean_signal,
    make_example,
    one_hot,
)

__all__ = [
    # Constants
    "CONTEXT_WINDOW", "FC_INPUT_SIZE", "FREQUENCIES", "N_FREQS",
    "N_TOTAL", "NOISE_LEVELS", "SAMPLE_RATE", "SEQ_FEATURES",
    # Signal primitives
    "add_gaussian_noise", "generate_clean_signal", "make_example", "one_hot",
    # Dataset
    "SignalReconstructionDataset", "get_dataloaders",
    # Models
    "FCNet", "LSTMNet", "RNNNet",
]
