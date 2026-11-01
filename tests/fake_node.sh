#!/usr/bin/env bash
set -euo pipefail
workspace_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$workspace_dir"
exec "$workspace_dir/.venv/bin/python" "$workspace_dir/tests/fake_node.py" "$@"
