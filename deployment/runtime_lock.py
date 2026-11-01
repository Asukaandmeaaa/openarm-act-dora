"""Portable source pins plus a machine-local seal of installed binaries."""
import importlib.metadata as metadata
import json,subprocess,sys
import re
from pathlib import Path
from .common import ROOT,digest


def verify_portable():
 lock=json.loads((ROOT/'sources.lock.json').read_text())
 if sys.version_info[:2]!=(3,12):raise ValueError('This release is locked to Python 3.12')
 for repo in lock['repositories']:
  def git(*args):return subprocess.check_output(['git','-C',str(ROOT/repo['path']),*args],text=True).strip()
  if git('rev-parse','HEAD')!=repo['commit']:raise ValueError('Source revision differs: '+repo['path'])
  if git('diff','--binary','--ignore-submodules=all','HEAD')!=(ROOT/repo['patch']).read_text().strip():raise ValueError('Source patch differs: '+repo['path'])
 for name,version in lock['packages'].items():
  installed=metadata.version(name)
  if (installed.split('+')[0] if name in ('torch','torchvision') else installed)!=version:raise ValueError(f'Package differs: {name} {installed} != {version}')
 dependency_locks=[ROOT/'requirements/runtime.lock']
 if metadata.version('torch').endswith('+cu128'):dependency_locks.append(ROOT/'requirements/cuda128.lock')
 for line in '\n'.join(p.read_text() for p in dependency_locks).splitlines():
  match=re.fullmatch(r'([A-Za-z0-9_.-]+)==([^ ;]+)',line)
  if match and metadata.version(match[1])!=match[2]:raise ValueError('Locked dependency differs: '+match[1])
 for relative,expected in lock['algorithm_files'].items():
  p=Path(metadata.distribution('lerobot').locate_file(relative))
  if digest(p)!=expected:raise ValueError('Algorithm differs: '+relative)
 return lock


def artifact_paths():
 paths=[ROOT/'assets/gravity/libcollection_gravity.so']
 sdk=metadata.distribution('openarm-can')
 paths += [Path(sdk.locate_file(f)).resolve() for f in sdk.files if str(f).endswith('.so')]
 for target in list(paths):
  for line in subprocess.check_output(['ldd',str(target)],text=True).splitlines():
   if '=>' in line:
    value=line.split('=>',1)[1].strip().split()[0]
    if value.startswith('/'):paths.append(Path(value).resolve())
 return sorted(set(paths))


def local_key(path):return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

def seal():
 lock=verify_portable()
 value={'source_lock_sha256':digest(ROOT/'sources.lock.json'),
        'requirements_sha256':digest(ROOT/'requirements/runtime.lock'),
        'cuda_constraints_sha256':digest(ROOT/'requirements/cuda128.lock'),
        'python':sys.version,
        'packages':dict(sorted((d.metadata['Name'],d.version) for d in metadata.distributions() if d.metadata['Name'])),
        'runtime_files':{local_key(p):digest(p) for p in artifact_paths()}}
 (ROOT/'runtime.local.json').write_text(json.dumps(value,indent=2)+'\n')
 return value


def verify():
 verify_portable()
 path=ROOT/'runtime.local.json'
 if not path.is_file():raise ValueError('Missing local runtime seal. Run scripts/setup.sh first.')
 lock=json.loads(path.read_text())
 if lock['source_lock_sha256']!=digest(ROOT/'sources.lock.json'):raise ValueError('Source lock changed; rebuild and seal the environment')
 if lock['requirements_sha256']!=digest(ROOT/'requirements/runtime.lock'):raise ValueError('Dependency lock changed; rebuild and seal')
 if lock['cuda_constraints_sha256']!=digest(ROOT/'requirements/cuda128.lock'):raise ValueError('CUDA dependency lock changed; rebuild and seal')
 for name,version in lock['packages'].items():
  if metadata.version(name)!=version:raise ValueError('Installed package changed since setup: '+name)
 for relative,expected in lock['runtime_files'].items():
  if digest(ROOT/relative)!=expected:raise ValueError('Runtime artifact changed since setup: '+relative)
