#!/usr/bin/env python3
"""Capture reviewed upstream diffs without embedding this machine's paths.

Run after intentional source edits, then rebuild and seal the local environment.
Package/algorithm pins are deliberately not updated by this command.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    for repo in lock['repositories']:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(ROOT / repo['path']), *args])
        if git('rev-parse', 'HEAD').decode().strip() != repo['commit']:
            raise ValueError('Review revision changes explicitly in sources.lock.json: ' + repo['path'])
        # Do not strip: trailing blank context lines are part of valid unified diffs.
        (ROOT / repo['patch']).write_bytes(git('diff', '--binary', '--ignore-submodules=all', 'HEAD'))
    print('Captured upstream patches. Rebuild changed packages and run scripts/seal_environment.py.')


if __name__ == '__main__':
    main()
