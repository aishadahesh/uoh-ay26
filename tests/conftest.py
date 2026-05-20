"""Pytest configuration: add project root to sys.path once for all tests."""

import sys
from pathlib import Path

# Ensure 'from src.xxx import ...' works in every test file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
