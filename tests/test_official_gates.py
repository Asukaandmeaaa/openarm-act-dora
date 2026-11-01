"""Exercise the removed local thresholds using fake transport and sensor input."""
import io
import json
import socket
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import openarm_can
import pyarrow as pa
import pytest


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    monkeypatch.setattr(openarm_can, 'OpenArm', Mock(side_effect=AssertionError('no CAN')))


def test_observer_emits_old_and_skewed_samples_with_original_times(monkeypatch):
    from dora_openarm_observer import main as observer
    events=[]
    for name, stamp in [('arm_right', 0), ('arm_left', 1),
                        ('camera_ceiling', 2), ('camera_wrist_right', 900_000_000)]:
        events.append(dict(type='INPUT', id=name, value=None,
                           metadata={'source_monotonic_ns':stamp}))
    events += [dict(type='INPUT', id='command', value=pa.array(['start']), metadata={}),
               dict(type='INPUT', id='tick')]
    class Node:
        send_output=Mock()
        def __iter__(self): return iter(events)
    node=Node()
    monkeypatch.setattr(observer.dora, 'Node', lambda:node)
    monkeypatch.setattr(observer, '_build_output', lambda *args:pa.array([1]))
    monkeypatch.setattr('sys.argv', ['observer'])
    observer.main()
    node.send_output.assert_called_once()
    metadata=node.send_output.call_args.args[2]
    assert metadata['source_skew_ns']==900_000_000
    assert metadata['source_age_ns']>200_000_000


def test_policy_client_keeps_old_observation_and_full_prediction(monkeypatch,tmp_path):
    from dora_openarm_local_policy_server import main as client
    monkeypatch.setattr(client,'execution_settings',lambda:{'mode':'official_blend'})
    value=pa.array([{'id':1, 'position':[0.]*16}])
    events=[dict(type='INPUT', id='observation', value=value,
                 metadata=dict(observation_monotonic_ns=1, source_age_ns=1_000_000_000,
                               source_skew_ns=900_000_000))]
    class Node:
        send_output=Mock()
        def __iter__(self): return iter(events)
    class Transport(io.StringIO):
        def readline(self):
            return json.dumps(dict(interval=33_333_333,positions=np.zeros((50,16)).tolist()))
    node=Node()
    monkeypatch.setattr(client.dora, 'Node', lambda:node)
    monkeypatch.setattr(client, 'Log', lambda _:Mock())
    monkeypatch.setattr(client, 'session', lambda:tmp_path)
    client._main_dora(Transport(),tmp_path)
    node.send_output.assert_called_once()
    _, actions, metadata=node.send_output.call_args.args
    assert len(actions)==50 and metadata['observation_monotonic_ns']==1
    assert metadata['reset'] is True


def test_ready_reports_pose_and_ages_without_numeric_admission(monkeypatch,tmp_path):
    from deployment.cli import live_ready
    for side in ('right','left'):
        (tmp_path/f'{side}_status.json').write_text(json.dumps(dict(
            enabled=True, fault=None, stopped=False, active=False, monotonic_ns=0,
            q=[.2]*8, qd=[.1]*8)))
    for camera in ('ceiling','wrist_right'):
        (tmp_path/f'camera_{camera}.json').write_text('{"monotonic_ns":0}')
    # AF_UNIX only, never a CAN socket or a real policy server.
    path=str(tmp_path/'policy.sock')
    monkeypatch.setenv('SOCKET',path)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(path)
        live_ready(tmp_path)
        (tmp_path/'stop.request').touch()
        with pytest.raises(ValueError,match='already started/stopped'):
            live_ready(tmp_path)
