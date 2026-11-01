#!/usr/bin/env python3
"""Verify published pins, then seal this machine's compiled artifacts. No hardware."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deployment.runtime_lock import seal
if __name__=='__main__':
 result=seal();print(f"Sealed {len(result['runtime_files'])} local binary/dependency hashes; no hardware accessed.")
