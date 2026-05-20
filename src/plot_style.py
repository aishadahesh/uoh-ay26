"""Shared plotting constants."""

from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCIES = [1, 2, 5, 7]
FREQ_LABELS = ["1 Hz", "2 Hz", "5 Hz", "7 Hz"]
SAMPLE_RATE = 1000
MODEL_COLORS = {"FC": "steelblue", "RNN": "tomato", "LSTM": "seagreen"}
