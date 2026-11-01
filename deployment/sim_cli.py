"""Isolated simulation launcher; never writes the physical session pointer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from .common import ROOT, write_json
from .cli import verify_model, verify_sources


def main():
    parser=argparse.ArgumentParser(description='MuJoCo-only selected-checkpoint test; no CAN or physical cameras')
    parser.add_argument('stage',choices=['view','test'])
    stage=parser.parse_args().stage
    verify_sources(); verify_model(); (ROOT/'results').mkdir(exist_ok=True)
    path=Path(tempfile.mkdtemp(prefix='simulation_',dir=ROOT/'results'))
    (ROOT/'results/current_sim').write_text(str(path)+'\n')
    env={**os.environ,'ACT_SESSION':str(path),'SOCKET':str(path/'policy.sock'),
         'ACT_SIM_VIEWER':'1' if stage=='view' else '0','PYTHONPATH':str(ROOT),
         'MUJOCO_GL':'glfw' if stage=='view' else 'egl'}
    print('SIMULATION ONLY: '+str(path),flush=True)
    with (path/'policy_stdout.txt').open('w') as out:
        policy=subprocess.Popen([str(ROOT/'scripts/node'),'policy'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+90
            while not Path(env['SOCKET']).exists():
                if policy.poll() is not None:raise RuntimeError((path/'policy_stdout.txt').read_text())
                if time.monotonic()>deadline:raise TimeoutError('policy startup')
                time.sleep(.1)
            write_json(path/'start.request',dict(monotonic_ns=time.monotonic_ns(),reason='simulation only'))
            command=[str(ROOT/'.venv/bin/dora'),'run',str(ROOT/'dataflow-sim.yml')]
            if stage=='test':command+=['--stop-after','8s']
            result=subprocess.run(command,cwd=ROOT,env=env)
            if result.returncode:raise RuntimeError('simulation dataflow failed; inspect logs '+str(path))
            status=json.loads((path/'simulation_status.json').read_text())
            if stage=='test':
                assert status['frames']>2 and status['sim_time_s']>0 and status['policy_targets']>0
                (ROOT/'validation').mkdir(exist_ok=True)
                write_json(ROOT/'validation/simulation_smoke.json',dict(
                    passed=True,hardware_accessed=False,physical_validation_passed=False,
                    logs=str(path),status=status))
            print(json.dumps(status,indent=2))
        finally:
            if policy.poll() is None:policy.terminate()
            try:policy.wait(timeout=5)
            except subprocess.TimeoutExpired:policy.kill();policy.wait()


if __name__=='__main__':main()
