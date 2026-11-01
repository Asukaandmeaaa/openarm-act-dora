"""Native node integration; no CAN, motors or cameras are constructed."""
import json
from datetime import datetime, timezone
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import openarm_can
import openarm_driver
import pyarrow as pa
import pytest
import yaml

from deployment import control, record, native_robot
from deployment.config import settings


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch,tmp_path):
    monkeypatch.setattr(openarm_can,'OpenArm',Mock(side_effect=AssertionError('no CAN')))
    for module in (control,record,native_robot):
        monkeypatch.setattr(module,'session',lambda:tmp_path)


def test_operator_routing_preserves_targets_and_stop(tmp_path):
    q=pa.array([{'qpos':[.1]*8}]); metadata={'chunk_id':'test'}
    tick=dict(type='INPUT',id='tick')
    move=dict(type='INPUT',id='move_position',value=q,metadata=metadata)
    sent=[]
    class Node:
        def send_output(self,*args): sent.append(args)
        def __iter__(self):
            yield tick
            yield move
            assert not any(x[0]=='arm_command' for x in sent)
            (tmp_path/'enable.request').touch()
            yield tick
            yield tick
            yield move
            assert not any(x[0]=='move_position_right' for x in sent)
            (tmp_path/'start.request').touch()
            yield move
            (tmp_path/'stop.request').touch()
            yield move
            (tmp_path/'disable.request').touch()
            yield tick
            yield tick
    control.run(Node(),hardware=True)
    commands=[x[1][0].as_py() for x in sent if x[0]=='arm_command']
    assert commands==['start','hold','stop']
    targets=[x for x in sent if x[0]=='move_position_right']
    assert len(targets)==1 and targets[0][1] is q and targets[0][2] is metadata


def test_stopped_session_cannot_enable(tmp_path):
    (tmp_path/'home.request').touch(); (tmp_path/'stop.request').touch()
    node=Mock();node.__iter__=Mock(return_value=iter([dict(type='INPUT',id='tick')]))
    control.run(node,hardware=True)
    assert all(c.args[0]!='arm_command' for c in node.send_output.call_args_list)


def test_native_entrypoint_uses_official_driver_only_on_start(monkeypatch,tmp_path):
    from dora_openarm import main as official
    (tmp_path/'driver.yaml').write_text(yaml.safe_dump(settings()))
    driver=SimpleNamespace(start=Mock(return_value=True),stop=Mock(),
                           last_command_dispatch_timestamp_ns=None)
    factory=Mock(return_value=driver)
    class Node:
        send_output=Mock()
        def __iter__(self):
            factory.assert_not_called()
            yield dict(type='INPUT',id='command',value=pa.array(['start']),metadata={})
            factory.assert_called_once()
            # End the stream without a stop command: --stop must override STOP=false.
    owner=Mock();owner.__enter__=Mock();owner.__exit__=Mock(return_value=False)
    monkeypatch.setattr(native_robot,'CanOwnership',lambda *args:owner)
    monkeypatch.setattr(official.dora,'Node',Node)
    monkeypatch.setattr(openarm_driver,'SingleArmDriver',factory)
    monkeypatch.setattr(sys,'argv',['test'])
    monkeypatch.setenv('STOP','false')  # shell variables must not cause exit homing
    native_robot.main('right')
    assert factory.call_args.args[0]=='right_arm'
    driver.start.assert_called_once();driver.stop.assert_called_once()
    owner.__enter__.assert_called_once();owner.__exit__.assert_called_once()


def test_native_lock_failure_never_enters_official_node(monkeypatch,tmp_path):
    from dora_openarm import main as official
    (tmp_path/'driver.yaml').write_text(yaml.safe_dump(settings()))
    monkeypatch.setattr(native_robot,'CanOwnership',Mock(side_effect=RuntimeError('owned')))
    monkeypatch.setattr(native_robot,'request',Mock())
    main=Mock();monkeypatch.setattr(official,'main',main)
    monkeypatch.setattr(sys,'argv',['test'])
    with pytest.raises(RuntimeError,match='owned'): native_robot.main('right')
    main.assert_not_called()


def test_official_yaml_start_and_mit_targets(monkeypatch):
    from openarm_driver.driver import SingleArmDriver, _create_default_checker
    from openarm_driver.config import Config
    cfg=object.__new__(Config);cfg._config=settings()
    clock=[100.]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    driver=object.__new__(SingleArmDriver)
    driver.config=cfg;driver.arm_side='right_arm'
    driver.last_command=np.r_[np.full(7,.1),-.04]
    driver.last_command_time_s=99.99;driver.last_command_dispatch_timestamp_ns=None
    driver._safety_stop_reason=None;driver._last_safety_warning_time_s=None
    driver.joint_offsets=np.zeros(8);driver.num_mit_motors=8;driver.gripper_posforce=False
    driver.kps=np.array(settings()['control_gains']['kps'])
    driver.kds=np.array(settings()['control_gains']['kds'])
    driver.safety_checker=_create_default_checker('right_arm',cfg)
    driver.set_latest_state=Mock()
    sent=[]
    driver.openarm=SimpleNamespace(get_arm=lambda:SimpleNamespace(
        mit_control_all=lambda params:sent.append(params)))
    assert driver._on_start()
    assert len(sent)==500
    assert all(p.dq==0 and p.tau==0 for step in sent for p in step)
    np.testing.assert_allclose([p.q for p in sent[-1]],[0.]*7+[-.02])
    assert settings()['stop']['moves']==[]


def test_official_stopped_is_not_reported_as_disabled(tmp_path):
    (tmp_path/'disable.request').touch()
    events=[dict(type='INPUT',id='right_status',value=pa.array(['stopped']),metadata={})]
    record.run(events,{'right':Mock(),'left':Mock()})
    status=json.loads((tmp_path/'right_status.json').read_text())
    assert status['enabled'] is None
    assert status['disable_attempted'] and not status['disable_confirmed']


def test_recorder_handles_dora_datetime_metadata(tmp_path):
    stamp=datetime.now(timezone.utc)
    event=dict(type='INPUT',id='right_commanded_position',
               value=pa.array([{'qpos':[0.]*8}]),metadata={'timestamp':stamp,'chunk_id':'a'})
    record.run([event],{'right':Mock(),'left':Mock()})
    status=json.loads((tmp_path/'right_status.json').read_text())
    assert status['command_metadata']['timestamp']==str(stamp)
    assert status['command_metadata']['chunk_id']=='a'


def test_session_yaml_is_hashed(monkeypatch,tmp_path):
    from deployment import cli
    monkeypatch.setattr(cli,'code_manifest',lambda:{})
    (tmp_path/'driver.yaml').write_text('original')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        backend='official_native',files={},driver_sha256=cli.digest(tmp_path/'driver.yaml'))))
    cli.assert_unchanged(tmp_path)
    (tmp_path/'driver.yaml').write_text('changed')
    with pytest.raises(ValueError,match='YAML changed'):cli.assert_unchanged(tmp_path)
