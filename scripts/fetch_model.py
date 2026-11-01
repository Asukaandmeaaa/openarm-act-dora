#!/usr/bin/env python3
"""Download the pinned six-file ACT checkpoint, verify it and install atomically."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(directory, manifest):
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Manifest requires flat checkpoint filenames')
        if sha256(directory / name) != expected:
            raise ValueError('Checkpoint hash mismatch: ' + name)


def install(archive, destination, manifest):
    if sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'checkpoint'
        stage.mkdir()
        with tarfile.open(archive, 'r:gz') as bundle:
            members = bundle.getmembers()
            if len(members) != len(manifest['files']) or {m.name for m in members} != set(manifest['files']):
                raise ValueError('Unexpected archive contents')
            for member in members:
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError('Non-regular or unsafe archive member')
                with bundle.extractfile(member) as source, (stage / member.name).open('xb') as target:
                    shutil.copyfileobj(source, target)
        verify(stage, manifest)
        if destination.exists():
            raise FileExistsError('Preserve existing model before replacing it: ' + str(destination))
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded release archive')
    parser.add_argument('--verify', action='store_true', help='Verify installed files without network access')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'artifacts/model.json').read_text())
    destination = ROOT / manifest['destination']
    if args.verify or destination.exists():
        verify(destination, manifest)
        print('VERIFIED', destination)
        return
    if args.archive:
        install(args.archive, destination, manifest)
    else:
        with tempfile.TemporaryDirectory(prefix='openarm-download-') as temporary:
            archive = Path(temporary) / 'model.tar.gz'
            print('Downloading', manifest['url'], flush=True)
            with urllib.request.urlopen(manifest['url'], timeout=120) as response, archive.open('wb') as target:
                shutil.copyfileobj(response, target)
            install(archive, destination, manifest)
    print('INSTALLED and SHA256 verified:', destination)


if __name__ == '__main__':
    main()
