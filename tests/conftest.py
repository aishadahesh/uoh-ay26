"""Pytest configuration: add project root to sys.path once for all tests."""

import sys
from pathlib import Path

import pytest

from src.data_generator import get_dataloaders
from src.models import FCNet

# Ensure 'from src.xxx import ...' works in every test file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def tiny_loaders():
    """Tiny train/val/test split for fast tests (200 samples total)."""
    return get_dataloaders(n_samples=200, batch_size=32, seed=0)


@pytest.fixture(scope="module")
def tiny_model():
    return FCNet(hidden_size=32)
