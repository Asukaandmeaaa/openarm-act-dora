import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'upstream/dora-openarm-evaluation'

def deployment_config_path():
    value = Path(os.environ.get('OPENARM_DEPLOYMENT_CONFIG', 'assets/deployment.json'))
    return value if value.is_absolute() else ROOT/value

def deployment_config():
    return json.loads(deployment_config_path().read_text())

MODEL = ROOT/deployment_config()['model_path']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def session():
    path = Path(os.environ['ACT_SESSION']).resolve()
    if not path.is_dir() or path.parent != ROOT/'results':
        raise ValueError('invalid prepared ACT session')
    return path


def request(name, reason=None):
    path = session()/f'{name}.request'
    data = dict(monotonic_ns=time.monotonic_ns(), reason=reason)
    # Atomic exclusive publication; readers never see partially-written JSON.
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data))
    try:
        os.link(temporary,path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink()


class Log:
    def __init__(self, role):
        self.file = (session()/f'{role}.jsonl').open('x', buffering=1)

    def __call__(self, event, **fields):
        def default(v):
            if hasattr(v, 'tolist'): return v.tolist()
            if hasattr(v, 'as_py'): return v.as_py()
            return str(v)
        self.file.write(json.dumps(dict(event=event, wall_ns=time.time_ns(),
                                       monotonic_ns=time.monotonic_ns(), **fields),
                                   default=default, allow_nan=False)+'\n')


def code_manifest():
    paths = list((ROOT/'deployment').glob('*.py')) + list((ROOT/'scripts').glob('*'))
    paths += [ROOT/'dataflow.yml', ROOT/'dataflow-sim.yml', ROOT/'sources.lock.json', ROOT/'runtime.local.json', ROOT/'requirements/runtime.lock', ROOT/'requirements/cuda128.lock', ROOT/'artifacts/model.json']
    paths += list((ROOT/'assets').rglob('*')) + list(MODEL.glob('*'))
    paths += list((ROOT/'sim/scenes').rglob('*'))
    for relative in ['src/local_policy_server.py',
                     'nodes/dora-openarm/src/dora_openarm/main.py',
                     'nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py',
                     'nodes/dora-openarm-observer/src/dora_openarm_observer/main.py',
                     'nodes/dora-openarm-local-policy-server/src/dora_openarm_local_policy_server/main.py']:
        paths.append(UPSTREAM/relative)
    paths += list((ROOT/'upstream/openarm_driver/src/openarm_driver').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}
