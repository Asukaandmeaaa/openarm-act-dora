"""Exercise actual driver constructor/start/home/stop with SDK-shaped fakes.

No real OpenArm is constructed. Specs use the installed extension classes so
an invented Python SDK method fails here, rather than during hardware startup.
"""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import openarm_can as oa
import pytest
from openarm_driver import Config, SingleArmDriver
from openarm_driver.driver import _BUS_COUNTER_NAMES
from deployment import feedforward as ff
from deployment.config import settings


@pytest.fixture
def hardware(monkeypatch,tmp_path):
    original=oa.OpenArm
    sdk=Mock(spec_set=original);arm=Mock(spec_set=oa.ArmComponent)
    q=np.array([0.,.03,.01,.02,.01,0,0,-.019])
    motors=[]
    for i in range(8):
        m=Mock(spec_set=oa.Motor)
        m.get_position.side_effect=lambda i=i:float(q[i])
        m.get_velocity.return_value=0.;m.get_torque.return_value=0.
        m.get_state_tmos.return_value=25;m.get_state_trotor.return_value=25
        m.get_error_code.return_value=1;m.has_error.return_value=False
        motors.append(m)
    arm.get_motors.return_value=motors
    link=Mock(spec_set=oa.MotorLinkStats);link.commands_sent=0;link.responses=0
    arm.get_link_stats.return_value=link
    sdk.get_arm.return_value=arm
    sdk.get_unmatched_frames.return_value={};sdk.is_link_running.return_value=True
    bus=Mock(spec_set=oa.BusStatus)
    for name in _BUS_COUNTER_NAMES:
        counter=Mock(spec_set=oa.ErrorCounter);counter.count=0;counter.seconds_ago.return_value=0.
        setattr(bus,name,counter)
    sdk.get_bus_status.return_value=bus
    sent=[]
    def capture(params):
        sent.append(params);q[:]=[p.q for p in params]
        link.commands_sent+=1;link.responses+=1
    arm.mit_control_all.side_effect=capture
    factory=Mock(spec_set=original,return_value=sdk)
    monkeypatch.setattr(oa,'OpenArm',factory)
    monkeypatch.setenv('ACT_SESSION',str(tmp_path))
    monkeypatch.setattr(ff,'Log',lambda _:Mock())
    monkeypatch.setattr(ff,'request',lambda name,reason:(tmp_path/(name+'.request')).touch())
    clock=[100.]
    monkeypatch.setattr(ff.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(ff.time,'sleep',lambda dt:clock.__setitem__(0,clock[0]+dt))
    cfg=object.__new__(Config);cfg._config=settings()
    # These historical feedforward regressions explicitly opt in, independently
    # of the production deployment's native zero-feedforward selection.
    cfg._config['collection_feedforward']['enabled']=True
    return SimpleNamespace(sdk=sdk,factory=factory,q=q,sent=sent,clock=clock,cfg=cfg,path=tmp_path)


def test_real_driver_entire_lifecycle(hardware):
    h=hardware;driver=SingleArmDriver('right_arm',h.cfg)
    h.sdk.enable_all.assert_not_called();assert not h.sent
    assert driver.start();assert len(h.sent)==500
    np.testing.assert_allclose(driver.last_command,np.r_[np.zeros(7),-.02],atol=1e-9)
    assert all(p.dq==0 for command in h.sent for p in command)
    assert any(p.tau!=0 for p in h.sent[0][:7])
    q=np.r_[0,.1,0,.1,0,0,0,-.03];v=np.r_[np.full(7,.2),0]
    driver.feedforward.receive(q,dict(reference_sequence=1,reference_dt_ns=4_000_000,
        reference_velocity_json=json.dumps(v.tolist())))
    driver.feedforward.tick(driver);assert h.sent[-1][1].dq==.2
    h.clock[0]+=.002;count=len(h.sent);driver.feedforward.tick(driver)
    assert len(h.sent)==count  # no premature zero on the separate timer
    h.clock[0]+=.0021;driver.feedforward.tick(driver)
    assert all(p.dq==0 for p in h.sent[-1])
    h.q[1]+=.1;(h.path/'stop.request').touch();driver.feedforward.tick(driver)
    np.testing.assert_allclose([p.tau for p in h.sent[-1]][:7],driver.feedforward.gravity(np.r_[0,.2,0,.1,0,0,0]))
    driver.stop();h.sdk.disable_all.assert_called_once()
    count=len(h.sent);driver.feedforward.tick(driver);assert len(h.sent)==count
    assert not driver.started and not driver._command_enabled


def test_native_driver_lifecycle_ignores_velocity_and_sends_zero_feedforward(hardware):
    h=hardware
    h.cfg._config['collection_feedforward']['enabled']=False
    driver=SingleArmDriver('right_arm',h.cfg)
    h.sdk.enable_all.assert_not_called()
    assert getattr(driver,'feedforward',None) is None
    assert driver.start()
    assert len(h.sent)==500
    h.clock[0]+=.04
    target=driver.last_command.copy();target[0]+=.001
    assert driver.send_position(target,velocity=np.ones(8))
    assert all(p.dq==0 and p.tau==0 for command in h.sent for p in command)
    np.testing.assert_allclose([p.q for p in h.sent[-1]],target)
    driver.stop()
    h.sdk.disable_all.assert_called_once()


def test_stop_before_enable_never_enables(hardware):
    h=hardware;driver=SingleArmDriver('right_arm',h.cfg)
    (h.path/'stop.request').touch()
    assert not driver.start()
    h.sdk.enable_all.assert_not_called();assert not h.sent


def test_interrupted_home_keeps_gravity_hold_until_disable(hardware,monkeypatch):
    h=hardware;driver=SingleArmDriver('right_arm',h.cfg)
    def sleep(dt):
        h.clock[0]+=dt
        if len(h.sent)==3:(h.path/'stop.request').touch()
    monkeypatch.setattr(ff.time,'sleep',sleep)
    assert not driver.start()
    assert not driver.started and driver._command_enabled
    last=driver.last_command.copy();h.q[1]+=.1;count=len(h.sent)
    driver.feedforward.tick(driver)
    assert len(h.sent)==count+1
    np.testing.assert_array_equal([p.q for p in h.sent[-1]],last)
    assert all(p.dq==0 for p in h.sent[-1])
    driver.stop();h.sdk.disable_all.assert_called_once()


def test_failed_stop_trajectory_still_attempts_disable(hardware):
    h=hardware;driver=SingleArmDriver('right_arm',h.cfg)
    assert driver.start()
    driver._on_stop=Mock(side_effect=RuntimeError('stop trajectory error'))
    with pytest.raises(RuntimeError,match='stop trajectory error'):driver.stop()
    h.sdk.disable_all.assert_called_once();assert not driver._command_enabled


def test_invalid_reference_still_uses_current_gravity(hardware):
    h=hardware;driver=SingleArmDriver('right_arm',h.cfg);assert driver.start()
    h.q[1]=.4;driver.set_latest_state();expected=driver.feedforward.gravity(h.q[:7])
    assert not driver.send_position(np.array([0,float('nan'),0,0,0,0,0,-.02]))
    np.testing.assert_allclose([p.tau for p in h.sent[-1]][:7],expected)
    assert all(p.dq==0 for p in h.sent[-1])
