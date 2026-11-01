#!/usr/bin/env bash
set -euo pipefail
install_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$install_root"
backend="${1:-cpu}"
case "$backend" in cpu|cu128) ;; *) echo 'Usage: scripts/install_python.sh [cpu|cu128]' >&2; exit 2;; esac
python3 -c 'import sys,platform; assert sys.version_info[:2]==(3,12), "Use Python 3.12"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Supported target: Linux x86_64"'
python3 -m venv .venv
.venv/bin/python -m pip install 'pip==26.0.1'
if [ "$backend" = cpu ]; then
  # All CPU transitive requirements are already in the common lock.
  .venv/bin/python -m pip install --no-deps 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install -c requirements/runtime.lock -c requirements/cuda128.lock \
    'torch==2.11.0' 'torchvision==0.26.0' \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
fi
.venv/bin/python -m pip install --no-deps -r requirements/runtime.lock
