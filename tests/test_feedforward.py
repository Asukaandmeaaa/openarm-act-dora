"""Real MIT parameter construction + KDL with a fake transport; no hardware."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import openarm_can
import pytest
import pyarrow as pa
from openarm_driver import Config, SingleArmDriver
from deployment import feedforward as ff
from deployment.config import settings
from deployment.gravity import Gravity


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch,tmp_path):
    monkeypatch.setattr(openarm_can,'OpenArm',Mock(side_effect=AssertionError('no CAN')))
    monkeypatch.setattr(ff,'request',lambda name,reason: (tmp_path/(name+'.request')).touch())


@pytest.fixture
def rig(tmp_path,monkeypatch):
    cfg=object.__new__(Config); cfg._config=settings()
    driver=object.__new__(SingleArmDriver)
    driver.config=cfg; driver.arm_side='right_arm'; driver.started=True
    driver.last_command=np.r_[np.zeros(7),-.02]
    driver.last_command_time_s=0; driver.last_command_dispatch_timestamp_ns=None
    driver._safety_stop_reason=None; driver._last_safety_warning_time_s=None
    driver.joint_offsets=np.zeros(8); driver.num_mit_motors=8; driver.gripper_posforce=False
    driver.kps=cfg.get_default_kps(); driver.kds=cfg.get_default_kds()
    driver.latest_state={'qpos':np.r_[0,.5,.1,.2,.1,0,0,-.03],'qvel':np.zeros(8)}
    driver.set_latest_state=Mock()
    commands=[]
    driver.openarm=SimpleNamespace(get_arm=lambda:SimpleNamespace(mit_control_all=commands.append))
    driver.feedforward=ff.CollectionFeedforward('right_arm',cfg,path=tmp_path,log=Mock())
    driver.test_clock=[100.]
    monkeypatch.setattr(ff.time,'monotonic',lambda:driver.test_clock[0])
    return driver,commands


def reference(q,v,sequence=1):
    return dict(reference_sequence=sequence,reference_dt_ns=4_000_000,
                reference_velocity_json=json.dumps(list(v)))


def test_real_mit_q_velocity_gravity_and_gripper(rig):
    driver,sent=rig
    q=np.array([.01,.2,.03,.1,.02,.01,.02,-.05]);v=np.r_[np.full(7,.1),0]
    driver.feedforward.receive(q,reference(q,v));driver.feedforward.tick(driver)
    np.testing.assert_array_equal([p.q for p in sent[-1]],q)
    np.testing.assert_array_equal([p.dq for p in sent[-1]],v)
    np.testing.assert_allclose([p.tau for p in sent[-1]],np.r_[Gravity('right')(driver.latest_state['qpos'][:7]),0],atol=1e-12)
    assert sent[-1][7].kp==5 and sent[-1][7].kd==.1
    # No next reference: zero dq while continuing gravity at the current pose.
    driver.latest_state['qpos'][1]+=.1
    previous_tau=sent[-1][1].tau
    driver.test_clock[0]+=.00401
    driver.feedforward.tick(driver)
    assert all(p.dq==0 for p in sent[-1])
    assert sent[-1][1].tau!=previous_tau
    np.testing.assert_array_equal([p.q for p in sent[-1]],q)


def test_stop_discards_queued_target_and_never_restores_velocity(rig,tmp_path):
    driver,sent=rig;q=np.r_[0,.1,0,.1,0,0,0,-.03];v=np.r_[np.full(7,.2),0]
    driver.feedforward.receive(q,reference(q,v));driver.feedforward.tick(driver)
    driver.feedforward.receive(q+.001,reference(q,v,2))
    (tmp_path/'stop.request').touch()
    driver.feedforward.tick(driver)
    driver.feedforward.receive(q+.002,reference(q,v,3));driver.feedforward.tick(driver)
    np.testing.assert_array_equal([p.q for p in sent[-1]],q)
    assert all(p.dq==0 for p in sent[-1]); assert driver.feedforward.held
    count=len(sent);(tmp_path/'disable.request').touch();driver.feedforward.tick(driver)
    assert len(sent)==count


@pytest.mark.parametrize('invalid', ['nan_position','inf_position','nan_velocity'])
def test_nonfinite_references_still_hold(rig,tmp_path,invalid):
    driver,sent=rig;initial=driver.last_command.copy();q=initial.copy();v=np.zeros(8)
    if invalid=='nan_position':q[0]=float('nan')
    if invalid=='inf_position':q[0]=float('inf')
    if invalid=='nan_velocity':v[0]=float('nan')
    driver.feedforward.receive(q,reference(q,v));driver.feedforward.tick(driver)
    np.testing.assert_array_equal([p.q for p in sent[-1]],initial)
    assert all(p.dq==0 for p in sent[-1])
    assert (tmp_path/'stop.request').exists()


def test_left_hold_recomputes_gravity_without_policy_motion(rig,tmp_path):
    driver,sent=rig;driver.arm_side='left_arm'
    driver.feedforward=ff.CollectionFeedforward('left_arm',driver.config,path=tmp_path,log=Mock())
    driver.feedforward.tick(driver)
    np.testing.assert_array_equal([p.q for p in sent[-1]],np.r_[np.zeros(7),-.02])
    assert all(p.dq==0 for p in sent[-1])
    np.testing.assert_allclose([p.tau for p in sent[-1]],np.r_[Gravity('left')(driver.latest_state['qpos'][:7]),0])


def test_differentiate_before_drops_reset_and_end():
    ref=ff.ReferenceVelocity();q=np.zeros(8)
    def attach(q,**extra):
        return ref.attach('right',q,dict(reference_dt_ns=4_000_000,**extra))
    assert json.loads(attach(q)['reference_velocity_json'])==[0]*8
    attach(q+.004)  # this transport sample may be dropped after differentiation
    result=attach(q+.008)
    np.testing.assert_allclose(json.loads(result['reference_velocity_json']),[1]*7+[0])
    assert json.loads(attach(q,reference_reset=True)['reference_velocity_json'])==[0]*8
    assert json.loads(attach(q+.004,reference_end=True)['reference_velocity_json'])==[0]*8


def test_gripper_keeps_existing_endpoint_control(rig):
    driver,sent=rig;q=driver.last_command.copy();q[7]=0
    assert driver.send_position(q)
    assert sent[-1][7].q==0 and sent[-1][7].dq==sent[-1][7].tau==0
    np.testing.assert_array_equal([p.q for p in sent[-1]][:7],q[:7])


def test_recorded_release_boundary_does_not_stop_and_can_reenter(rig,tmp_path):
    driver,sent=rig
    # Exact last accepted and first rejected targets in trial_dtkjfl58.
    driver.last_command=np.array([.9702690243721008,-.17447486519813538,
        -.8077834248542786,.5636284351348877,-.5795983672142029,
        .539000928401947,.9941577911376953,-.07398897409439087])
    q=np.array([.9703261852264404,-.17457303404808044,-.8079453110694885,
        .5634539127349854,-.5793638825416565,.538934588432312,
        .9939431548118591,-.07394051551818848])
    v=np.r_[.01,-.03,.02,.01,.02,.01,.01,0]
    assert driver.send_position(q,velocity=v)
    assert sent[-1][1].q==-.17453 and sent[-1][1].dq==0
    np.testing.assert_array_equal(np.array([p.q for p in sent[-1]])[[0,2,3,4,5,6,7]],q[[0,2,3,4,5,6,7]])
    assert not driver.feedforward.held and not (tmp_path/'stop.request').exists()
    record=driver.feedforward.log.call_args.kwargs
    assert record['position_limited_axes']==[2]
    np.testing.assert_array_equal(record['requested_position'],q)
    np.testing.assert_array_equal(record['requested_velocity'],v)
    q[1]=-.17;v[1]=.05
    assert driver.send_position(q,velocity=v)
    assert sent[-1][1].dq==.05 and driver.feedforward.position_limited_axes==[]


def test_exact_endpoint_never_has_outward_feedforward(rig):
    driver,sent=rig;q=driver.last_command.copy();q[1]=-.17453
    v=np.zeros(8);v[1]=-.1
    assert driver.send_position(q,velocity=v)
    assert sent[-1][1].q==-.17453 and sent[-1][1].dq==0


def test_finite_boundary_and_speed_saturate_without_stop(rig,tmp_path):
    driver,sent=rig;q=driver.last_command.copy();q[1]=-.175
    v=np.zeros(8);v[1]=-100;v[2]=5.704246461391449
    assert driver.send_position(q,velocity=v)
    assert sent[-1][1].q==-.17453 and sent[-1][1].dq==0
    assert sent[-1][2].dq==pytest.approx(5.4454)
    assert driver.feedforward.velocity_limited_axes==[2,3]
    assert not (tmp_path/'stop.request').exists()


def test_recorded_j7_crossing_and_return_keep_executing(rig,tmp_path):
    driver,sent=rig
    data=json.loads((Path(__file__).parent/'fixtures/j7_boundary_trial_wluriogi.json').read_text())
    driver.last_command=np.array(data['previous'])
    clipped=0
    for frame in data['frames']:
        q=np.array(frame['q']);v=np.array(frame['v'])
        assert driver.send_position(q,velocity=v)
        actual=np.array([p.q for p in sent[-1]])
        expected=np.clip(q,driver.feedforward.limits[:,0],driver.feedforward.limits[:,1])
        np.testing.assert_array_equal(actual,expected)
        if q[6]>1.5708:
            clipped+=1
            assert sent[-1][6].dq==0
            assert 7 in driver.feedforward.position_limited_axes
        assert not driver.feedforward.held
    assert clipped==0 and sent[-1][6].q<1.4
    assert not (tmp_path/'stop.request').exists()
    assert not (tmp_path/'right_fault.json').exists()


@pytest.mark.parametrize('prefetch_ms',[0,150])
def test_real_recorded_fifty_step_release_reaches_mit_open_target(rig,tmp_path,prefetch_ms):
    import asyncio
    from deployment.chunk_executor import execute_chunk
    from dora_openarm_actions_executor.main import BiquadLowpass
    driver,sent=rig
    prediction=json.loads((Path(__file__).parent/'fixtures/release50_trial_phio1fa8.json').read_text())['prediction']
    driver.last_command=np.array(prediction['state'][:8])
    driver.latest_state['qpos']=driver.last_command.copy()
    event=dict(value=pa.array(prediction['positions'],type=pa.list_(pa.float32())),
               metadata=dict(interval=33_333_333,reset=True,chunk_id='saved-release'))
    clock=[0];rv=ff.ReferenceVelocity();acks=[]
    class Node:
        def send_output(self,name,value,metadata):
            if name=='chunk_done':acks.append(metadata);return
            if name!='move_position_right':return
            q=value.field('qpos')[0].as_py()
            metadata=rv.attach(name,q,metadata)
            driver.test_clock[0]=100+clock[0]/1e9
            driver.feedforward.receive(q,metadata)
            driver.feedforward.tick(driver)
    async def sleep(delay):clock[0]+=round(delay*1e9)
    assert asyncio.run(execute_chunk(Node(),event,50,BiquadLowpass(250,15),
        lambda:False,Mock(),now=lambda:clock[0],sleep=sleep,prefetch_ms=prefetch_ms))
    assert len(sent)==417 and len(acks)==1
    assert sent[-1][7].q<-.44 and sent[-1][7].dq==sent[-1][7].tau==0
    assert all(p.dq==0 for p in sent[-1])
    assert not driver.feedforward.held and not (tmp_path/'stop.request').exists()


@pytest.mark.parametrize('transition_ms',[0,100,150])
def test_prefetched_saved_release_reaches_real_mit_constructor(rig,tmp_path,transition_ms):
    import asyncio
    from deployment.pipeline_execution import execute_pipeline_chunk
    from dora_openarm_actions_executor.main import BiquadLowpass
    driver,sent=rig
    prediction=json.loads((Path(__file__).parent/'fixtures/release50_trial_phio1fa8.json').read_text())['prediction']
    previous=np.array(prediction['state'])
    driver.last_command=previous[:8].copy()
    driver.latest_state['qpos']=driver.last_command.copy()
    event=dict(value=pa.array(prediction['positions'],type=pa.list_(pa.float32())),
        metadata=dict(interval=33_333_333,reset=False,chunk_id='1',observation_monotonic_ns=0))
    clock=[150_000_000];rv=ff.ReferenceVelocity();acks=[]
    class Node:
        def send_output(self,name,value,metadata):
            if name=='chunk_done':acks.append(metadata);return
            if name!='move_position_right':return
            q=value.field('qpos')[0].as_py()
            metadata=rv.attach(name,q,metadata)
            driver.test_clock[0]=100+clock[0]/1e9
            driver.feedforward.receive(q,metadata)
            driver.feedforward.tick(driver)
    async def sleep(delay):clock[0]+=round(delay*1e9)
    _,completed=asyncio.run(execute_pipeline_chunk(Node(),event,
        dict(prefetch_ms=150,transition_ms=transition_ms),BiquadLowpass(250,15),previous,
        lambda:False,Mock(),now=lambda:clock[0],sleep=sleep))
    assert completed and len(sent)==380 and len(acks)==1
    assert sent[-1][7].q<-.45 and sent[-1][7].dq==sent[-1][7].tau==0
    assert all(p.dq==0 for p in sent[-1])
    assert not driver.feedforward.held and not (tmp_path/'stop.request').exists()


@pytest.mark.parametrize('target',[1.1,4.,13.])
def test_finite_large_step_only_uses_urdf_position_range(rig,tmp_path,target):
    driver,sent=rig;q=driver.last_command.copy();q[0]=target
    assert driver.send_position(q,velocity=np.zeros(8))
    assert sent[-1][0].q==min(target,3.4907)
    assert not (tmp_path/'stop.request').exists()


def test_finite_gravity_saturates_without_latching_stop(rig,tmp_path):
    driver,sent=rig
    driver.feedforward.gravity=Mock(return_value=np.full(7,100.))
    assert driver.send_position(driver.last_command.copy())
    np.testing.assert_array_equal([p.tau for p in sent[-1]],[40,40,27,27,7,7,7,0])
    assert driver.feedforward.torque_limited_axes==list(range(1,8))
    assert not (tmp_path/'stop.request').exists()


def test_recorded_chunk25_velocity_fault_is_now_saturation_only(rig,tmp_path):
    driver,sent=rig
    data=json.loads((Path(__file__).parent/'fixtures/velocity_fault_trial_rp4q63ur.json').read_text())
    driver.last_command=np.array(data['previous']['q_ref'])
    failed=data['rejected']
    driver.latest_state['qpos']=np.array(failed['q_feedback'])
    assert driver.send_position(np.array(failed['requested_position']),
                                velocity=np.array(failed['requested_velocity']))
    np.testing.assert_array_equal([p.q for p in sent[-1]],failed['requested_position'])
    assert sent[-1][2].dq==pytest.approx(5.4454)
    assert driver.feedforward.velocity_limited_axes==[3]
    assert not driver.feedforward.held and not (tmp_path/'stop.request').exists()
    q=np.array(failed['requested_position']);q[2]+=.001
    assert driver.send_position(q,velocity=np.zeros(8))
    assert sent[-1][2].q==q[2]

@pytest.mark.parametrize('bad',[float('nan'),-1,[1]*6,[1,1,1,1,1,1,float('inf')]])
def test_restored_mode_rejects_non_scalar_one_gravity(rig,tmp_path,bad):
    driver,_=rig
    driver.config._config['collection_feedforward']['gravity_scale']=bad
    with pytest.raises(ValueError,match='frozen gravity scale 1.0'):
        ff.CollectionFeedforward('right_arm',driver.config,path=tmp_path,log=Mock())



def test_gravity_failure_clears_velocity_at_last_target(rig):
    driver,sent=rig;q=driver.last_command.copy();v=np.r_[np.full(7,.1),0]
    driver.send_position(q,velocity=v);old_tau=np.array([p.tau for p in sent[-1]])
    driver.feedforward.gravity=Mock(side_effect=RuntimeError('bad gravity'))
    assert not driver.send_position(q,velocity=v)
    assert all(p.dq==0 for p in sent[-1])
    np.testing.assert_array_equal([p.tau for p in sent[-1]],old_tau)


def test_node_routes_reference_tick_hold_disable_and_cleanup(rig):
    from dora_openarm import main as node
    driver,sent=rig;driver.start=Mock(return_value=True);driver.stop=Mock()
    q=np.r_[0,.1,0,.1,0,0,0,-.03];v=np.r_[np.full(7,.2),0]
    def event(name,value=None,metadata=None):
        return dict(type='INPUT',id=name,value=value,metadata=metadata or {})
    class Node:
        send_output=Mock()
        def __iter__(self):
            yield event('command',pa.array(['start']))
            yield event('move_position',pa.array([{'qpos':q}]),reference(q,v))
            assert len(sent)==1
            yield event('control_tick')
            assert len(sent)==1  # timer phase must not overwrite valid velocity
            assert any(p.dq for p in sent[-1])
            yield event('command',pa.array(['hold']))
            assert all(p.dq==0 for p in sent[-1])
            yield event('move_position',pa.array([{'qpos':q+.001}]),reference(q,v,2))
            yield event('control_tick')
            np.testing.assert_allclose([p.q for p in sent[-1]],q,atol=1e-8)
            yield event('command',pa.array(['stop']))
            count=len(sent);yield event('control_tick');assert len(sent)==count
    node.run(Node(),node.parse_args(['--side','right','--no-align']),lambda _:driver)
    driver.stop.assert_called_once()


def test_gravity_load_error_happens_before_can(monkeypatch):
    cfg=object.__new__(Config);cfg._config=settings()
    cfg._config['collection_feedforward']['enabled']=True
    monkeypatch.setattr(ff,'Gravity',Mock(side_effect=ValueError('bad URDF')))
    with pytest.raises(ValueError,match='bad URDF'):SingleArmDriver('right_arm',cfg)
    openarm_can.OpenArm.assert_not_called()


def test_home_uses_gravity_and_zero_velocity_and_stop_interrupts(rig,monkeypatch,tmp_path):
    import time
    driver,sent=rig;driver.last_command=np.r_[np.full(7,.03),-.019]
    def sleep(_):
        if len(sent)==3:(tmp_path/'stop.request').touch()
    monkeypatch.setattr(time,'sleep',sleep)
    assert driver._on_start() is False
    assert len(sent)==4
    assert all(p.dq==0 for command in sent for p in command)
    assert sent[0][1].tau!=0
    np.testing.assert_array_equal([p.q for p in sent[-1]],[p.q for p in sent[-2]])


def test_node_exception_always_calls_driver_stop(rig):
    from dora_openarm import main as node
    driver,_=rig;driver.start=Mock(return_value=True);driver.stop=Mock()
    class Node:
        send_output=Mock()
        def __iter__(self):
            yield dict(type='INPUT',id='command',value=pa.array(['start']),metadata={})
            raise RuntimeError('broken transport')
    with pytest.raises(RuntimeError,match='broken transport'):
        node.run(Node(),node.parse_args(['--side','right','--no-align']),lambda _:driver)
    driver.stop.assert_called_once()


def test_node_reports_fault_before_cleanup(rig,tmp_path):
    from dora_openarm import main as node
    driver,_=rig;driver.start=Mock(return_value=True)
    def stop(): assert (tmp_path/'stop.request').exists()
    driver.stop=Mock(side_effect=stop)
    class Node:
        send_output=Mock()
        def __iter__(self):
            yield dict(type='INPUT',id='command',value=pa.array(['start']),metadata={})
            raise RuntimeError('broken transport')
    with pytest.raises(RuntimeError,match='broken transport'):
        node.run(Node(),node.parse_args(['--side','right','--no-align']),lambda _:driver)
    driver.stop.assert_called_once()


def test_node_does_not_repeat_failed_disable(rig):
    from dora_openarm import main as node
    driver,_=rig;driver.start=Mock(return_value=True)
    driver.stop=Mock(side_effect=RuntimeError('disable transport failed'))
    events=[dict(type='INPUT',id='command',value=pa.array([command]),metadata={})
            for command in ('start','stop')]
    class Node:
        send_output=Mock()
        def __iter__(self):return iter(events)
    with pytest.raises(RuntimeError,match='disable transport failed'):
        node.run(Node(),node.parse_args(['--side','right','--no-align']),lambda _:driver)
    driver.stop.assert_called_once()
