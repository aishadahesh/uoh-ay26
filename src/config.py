"""Shared constants for the signal reconstruction experiment."""

from __future__ import annotations

FREQUENCIES: list[int] = [1, 2, 5, 7]
N_FREQS: int = len(FREQUENCIES)

SAMPLE_RATE: int = 1_000
DURATION: int = 10
N_TOTAL: int = SAMPLE_RATE * DURATION

# 100 samples @ 1 kHz = 100 ms window.
# This is the minimum needed for the mixed-signal component extraction task:
#   7 Hz → 0.7 cycles visible  (clear oscillation; easiest to separate)
#   5 Hz → 0.5 cycles visible  (half-sine visible)
#   2 Hz → 0.2 cycles          (slow ramp — challenging)
#   1 Hz → 0.1 cycles          (nearly flat — hardest; most sensitive to the mixture)
CONTEXT_WINDOW: int = 100

# FC  : flat vector  [noisy_window(100) | C(4) | sigma(1)] = 105
# Seq : per-step     [noisy_val, C1, C2, C3, C4, sigma]    = 6
FC_INPUT_SIZE: int = CONTEXT_WINDOW + N_FREQS + 1   # 105
SEQ_FEATURES: int  = 1 + N_FREQS + 1               # 6

NOISE_LEVELS: list[float] = [0.0, 0.1, 0.3, 0.5, 1.0]
FREQ_LABELS: list[str] = ["1 Hz", "2 Hz", "5 Hz", "7 Hz"]

MODEL_COLORS: dict[str, str] = {
    "FC": "steelblue",
    "RNN": "tomato",
    "LSTM": "seagreen",
}
