#!/usr/bin/env bash
# Install dependencies and build pure math/SDK bindings. Never starts hardware.
set -euo pipefail
setup_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$setup_root"
backend="${1:-cpu}"
case "$backend" in cpu|cu128) ;; *) echo 'Usage: scripts/setup.sh [cpu|cu128]' >&2; exit 2;; esac
if [ "${2:-}" = "--skip-python" ]; then
  test -x .venv/bin/python
elif [ "$#" -le 1 ]; then
  ./scripts/install_python.sh "$backend"
else
  echo 'Usage: scripts/setup.sh [cpu|cu128]' >&2; exit 2
fi
.venv/bin/python scripts/bootstrap_sources.py
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}" .venv/bin/python -m pip install --no-build-isolation --no-deps ./upstream/openarm_can/python
.venv/bin/python -m pip install --no-build-isolation --no-deps -e ./upstream/openarm_driver
for node in dora-openarm dora-openarm-actions-executor dora-openarm-observer dora-openarm-local-policy-server; do
  .venv/bin/python -m pip install --no-build-isolation --no-deps -e "./upstream/dora-openarm-evaluation/nodes/$node"
done
.venv/bin/python scripts/build_gravity.py
.venv/bin/python scripts/seal_environment.py
.venv/bin/python -m pip check
printf '%s\n' 'Setup complete. Next: python3 scripts/fetch_model.py; ./scripts/actctl check' 'No cameras or motors were opened or enabled.'
