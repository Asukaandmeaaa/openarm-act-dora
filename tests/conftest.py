"""Regression tests use the published hardware fixture, not operator overrides."""
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def published_hardware_fixture(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('OPENARM_HARDWARE_CONFIG', str(root / 'assets/hardware.yaml'))
