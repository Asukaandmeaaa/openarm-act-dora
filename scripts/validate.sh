#!/usr/bin/env bash
# Software-only: no production dataflow, CAN, motor SDK construction or cameras.
set -euo pipefail
validation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$validation_root"
python3 scripts/fetch_model.py
./scripts/actctl check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
HF_HUB_OFFLINE=1 .venv/bin/python scripts/check_model.py
