"""Operator entrypoint. No hardware imports or I/O in prepare/check/status."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import yaml
from .common import ROOT, MODEL, digest, code_manifest, write_json, request, deployment_config, deployment_config_path
from .config import settings


def mode():
    return deployment_config()['control_mode']


def require_mode():
    if mode() not in ('official_position','collection_feedforward'):
        raise ValueError('physical baseline control mode awaits operator choice; enable blocked')
    configured=settings().get('collection_feedforward',{}).get('enabled',False)
    if configured != (mode()=='collection_feedforward'):
        raise ValueError('deployment mode and hardware YAML differ')


def verify_feedforward():
    require_mode()
    if mode()=='collection_feedforward':
        from .gravity import Gravity
        import numpy as np
        if settings()['collection_feedforward'].get('gravity_scale') != 1.0:
            raise ValueError('restored collection mode requires gravity scale 1.0')
        for side in ('right','left'):
            gravity=Gravity(side)
            if not np.isfinite(gravity(np.zeros(7))).all():
                raise ValueError('gravity offline check failed')
            gravity.close()


def verify_model():
    from .chunk_execution import execution_settings
    execution=execution_settings()
    config=json.loads((MODEL/'config.json').read_text())
    expected={'observation.state':[16], 'observation.images.ceiling':[3,480,640],
              'observation.images.wrist_right':[3,480,640]}
    assert {k:v['shape'] for k,v in config['input_features'].items()} == expected
    assert config['type']=='act' and config['chunk_size']==execution.get('chunk_size',50)
    assert config['output_features']['action']['shape']==[16]
    for name in ('model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        if not (MODEL/name).is_file(): raise ValueError('missing checkpoint file '+name)
    deployment=deployment_config()
    for name, expected_hash in deployment.get('model_files_sha256',{}).items():
        if digest(MODEL/name) != expected_hash:
            raise ValueError('model/processor differs from deployment pin: '+name)
    if execution['mode']=='act_temporal':
        post=json.loads((MODEL/'policy_postprocessor.json').read_text())['steps']
        if ([s['registry_name'] for s in post]!=['unnormalizer_processor','device_processor']
                or post[0]['config']['norm_map']['ACTION']!='MEAN_STD'):
            raise ValueError('denormalized temporal ensemble requires frozen affine action postprocessor')


def verify_sources():
    from .runtime_lock import verify
    verify()


def load_session():
    path=Path((ROOT/'results/current').read_text().strip())
    if path.parent != ROOT/'results' or not path.is_dir(): raise ValueError('invalid session pointer')
    os.environ['ACT_SESSION']=str(path)
    os.environ['SOCKET']=str(Path(tempfile.gettempdir())/('dora-act-'+str(os.getuid())+'-'+path.name+'.sock'))
    return path


def assert_unchanged(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('backend')!='official_native':
        raise ValueError('previous deployment session; stop old processes and prepare a new session')
    recorded=manifest['files']
    if recorded != code_manifest(): raise ValueError('code/model changed since prepare; prepare a fresh session')
    if digest(path/'driver.yaml')!=manifest['driver_sha256']:
        raise ValueError('session hardware YAML changed since prepare')


def live_ready(path):
    if any((path/f'{s}.request').exists() for s in ('start','stop','disable')):
        raise ValueError('session already started/stopped; no automatic retry')
    now=time.monotonic_ns()
    for side in ('right','left'):
        if (path/f'{side}_fault.json').exists():
            raise ValueError('official node fault: '+(path/f'{side}_fault.json').read_text())
        state=json.loads((path/f'{side}_status.json').read_text())
        if not state['enabled'] or state['fault'] or state['stopped'] or state['active']:
            raise ValueError('arm startup incomplete or session stopped: '+side)
        if state['q'] is None or state['qd'] is None:
            raise ValueError('waiting for first official state output: '+side)
        print(f'{side}: q={state["q"]}, qd={state["qd"]}, '
              f'status_age_ms={(now-state["monotonic_ns"])/1e6:.1f}')
    for role in ('ceiling','wrist_right'):
        state=json.loads((path/f'camera_{role}.json').read_text())
        print(f'{role}: frame_age_ms={(now-state["monotonic_ns"])/1e6:.1f}')
    if not Path(os.environ['SOCKET']).is_socket(): raise ValueError('policy server not listening')


def main():
    parser=argparse.ArgumentParser(description='Independent official Dora ACT workspace; enable/home/start are physical actions.')
    parser.add_argument('stage',choices=['check','prepare-host','prepare','policy','dora','enable','home','ready','start','stop','disable','status'])
    stage=parser.parse_args().stage
    if stage=='prepare-host':
        from .host_setup import prepare_can_host
        try:
            prepare_can_host(settings()['can_interface'].values())
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            parser.exit(1, f'CAN host setup failed: {exc}\n')
        return
    if stage=='check':
        from .chunk_execution import execution_settings
        verify_sources(); verify_model(); verify_feedforward(); config=settings()
        print(json.dumps(dict(checkpoint=str(MODEL),control_mode=mode(),
                              action_execution=execution_settings(),
                              can_interfaces=config['can_interface'],hardware_accessed=False,
                              physical_validation_passed=False),indent=2)); return
    if stage=='prepare':
        from .chunk_execution import execution_settings
        verify_sources(); verify_model(); verify_feedforward(); (ROOT/'results').mkdir(exist_ok=True)
        # Do not replace the session pointer while an existing Dora process lives.
        current=ROOT/'results/current'
        if current.exists():
            old=Path(current.read_text().strip())
            for role in ('dora','policy'):
                marker=old/f'{role}_process.json'
                if marker.exists():
                    record=json.loads(marker.read_text()); pid=record['pid']
                    if Path(f'/proc/{pid}').exists(): raise ValueError('stop previous '+role+' before prepare')
        path=Path(tempfile.mkdtemp(prefix='trial_',dir=ROOT/'results'))
        (path/'driver.yaml').write_text(yaml.safe_dump(settings(),sort_keys=False))
        write_json(path/'manifest.json',dict(files=code_manifest(),control_mode=mode(),
                                           action_execution=execution_settings(),
                                           checkpoint=MODEL.parent.name,hardware_accessed_at_prepare=False,
                                           model_path=str(MODEL.relative_to(ROOT)),
                                           model_provenance=deployment_config().get('model_provenance'),
                                           deployment_config=str(deployment_config_path()),
                                           physical_validation_passed=False,
                                           backend='official_native',driver_sha256=digest(path/'driver.yaml')))
        current.write_text(str(path)+'\n')
        print('PREPARED '+str(path)); return
    path=load_session()
    if stage in ('policy','dora','enable','home','start'):
        verify_sources(); assert_unchanged(path)
    if stage in ('policy','dora'):
        if (path/f'{stage}_process.json').exists(): raise ValueError('this software stage already attempted; preserve logs and prepare fresh session')
        if stage=='dora':
            from .host_setup import require_can_host
            try:
                require_can_host(yaml.safe_load((path/'driver.yaml').read_text())['can_interface'].values())
            except RuntimeError as exc:
                parser.exit(1, f'CAN startup preparation failed: {exc}\n')
        if stage=='dora' and not Path(os.environ['SOCKET']).is_socket(): raise ValueError('wait for policy Listening')
        write_json(path/f'{stage}_process.json',dict(pid=os.getpid(),monotonic_ns=time.monotonic_ns()))
        command=([str(ROOT/'scripts/node'),'policy'] if stage=='policy' else
                 [str(ROOT/'.venv/bin/dora'),'run',str(ROOT/'dataflow.yml')])
        os.execvpe(command[0],command,os.environ)
    elif stage in ('enable','home'):
        require_mode()
        if any((path/f'{s}.request').exists() for s in ('enable','home','start','stop','disable')):
            raise ValueError('enable/home is one-shot; prepare a fresh session after a failure')
        if stage=='home':
            prompt=('Both arms supported, EMPTY grippers, '
                    'path to zero clear, old ROS/CAN writers stopped, emergency stop ready? '
                    'HOME enables BOTH arms and moves to zero over about 5 seconds. Type HOME: ')
        else:
            prompt=('Arms supported, empty grippers, path to zero clear, old ROS/CAN writers stopped, '
                    'emergency stop ready? ENABLE uses the official YAML to move BOTH arms toward zero over about 5 seconds. Type ENABLE: ')
        if input(prompt)!=stage.upper():
            print('Cancelled: no request sent; exact uppercase '+stage.upper()+' is required.')
            return
        request(stage)
        print('HOME requested: both arms will move to zero; ACT remains gated.' if stage=='home'
              else 'ENABLE requested: both arms will move toward zero; ACT awaits START.')
    elif stage=='ready':
        live_ready(path); print('READY: driver startup completed and camera samples exist; pose/ages are diagnostic, arrival is not verified. ACT has not started.')
    elif stage=='start':
        require_mode(); live_ready(path); request('start'); print('ACT START requested; monitor motion and logs.')
    elif stage=='stop':
        request('stop','operator')
        session_mode=json.loads((path/'manifest.json').read_text())['control_mode']
        if session_mode=='collection_feedforward':
            print('ACT STOP requested: zero reference velocity and hold last dispatched position with gravity updates. No power-off; NOT an emergency stop.')
        else:
            print('ACT targets blocked; the last motor target remains. No repeated hold commands, no power-off. NOT an emergency stop.')
    elif stage=='disable':
        if input('Both arms physically supported? Type DISABLE: ')!='DISABLE': return
        request('stop','operator_disable'); request('disable'); print('Official arm STOP requested. The official stopped status is not a disable acknowledgement; verify motor power state on site before exiting Dora.')
    else:
        for name in ('right_status.json','left_status.json','right_fault.json','left_fault.json'):
            if (path/name).exists(): print(name, (path/name).read_text())
        print('requests:',[p.name for p in path.glob('*.request')])


if __name__=='__main__': main()
