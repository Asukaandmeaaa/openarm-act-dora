"""No devices: genuine LeRobot ensemble, fake clocks and Arrow transport."""
import json
from unittest.mock import Mock
import numpy as np
import pyarrow as pa
import pytest
import torch
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from deployment.temporal_execution import TemporalController
from deployment.temporal_client import INTERVAL, run
from deployment.feedforward import ReferenceVelocity


def event(step, q=None, epoch=1_000_000_000):
    if q is None:
        q = np.full((50,16), step/100, dtype=np.float32)
    return dict(type='INPUT', id='actions',value=pa.array(q.tolist(),type=pa.list_(pa.float32())),
        metadata=dict(interval=INTERVAL, temporal_step=step, temporal_epoch_ns=epoch,
                      temporal_target_ns=epoch+(step+1)*INTERVAL,chunk_id=f'{epoch}:{step}'))


@pytest.mark.parametrize('coefficient',[0.,.01,-0.1047709150247644,-0.046041184561757126,0.02555339198176399,0.04743340266055522,0.06216167984441909,0.04192688834388196])
def test_real_lerobot_matches_timestamp_weighted_sum_including_zero_axes(coefficient):
    rng=np.random.default_rng(8)
    history=rng.normal(size=(120,50,16)).astype(np.float32)
    history[:,:,8:]=0  # Real left-arm zeros are valid actions, not empty slots.
    c=TemporalController(coefficient,Mock())
    for step,q in enumerate(history):
        e=event(step,q); t=e['metadata']['temporal_target_ns']
        c.accept(e,t-1); c.activate(t)
        contributors=range(max(0,step-49),step+1)
        aligned=np.array([history[i,step-i] for i in contributors])
        weights=np.exp(-coefficient*np.arange(len(aligned)))
        expected=np.average(aligned,axis=0,weights=weights)
        np.testing.assert_allclose(c.goal,expected,rtol=3e-5,atol=5e-7)
        assert c.count==min(50,step+1)


@pytest.mark.parametrize('coefficient',[.01,-0.1047709150247644])
def test_frozen_affine_postprocessor_commutes_with_ensemble(coefficient):
    from deployment.common import MODEL
    config=json.loads((MODEL/'policy_postprocessor.json').read_text())
    assert [s['registry_name'] for s in config['steps']]==['unnormalizer_processor','device_processor']
    assert config['steps'][0]['config']['norm_map']['ACTION']=='MEAN_STD'
    from safetensors.torch import load_file
    stats=load_file(str(MODEL/config['steps'][0]['state_file']))
    mean=stats['action.mean']; scale=stats['action.std']+config['steps'][0]['config']['eps']
    a=ACTTemporalEnsembler(coefficient,50); b=ACTTemporalEnsembler(coefficient,50)
    generator=torch.Generator().manual_seed(11)
    for _ in range(60):
        q=torch.randn((1,50,16),generator=generator)
        torch.testing.assert_close(a.update(q)*scale+mean,b.update(q*scale+mean))


def test_gap_resets_instead_of_fusing_nonadjacent_predictions():
    log=Mock(); c=TemporalController(.01,log)
    for step in (0,1,6,7):
        e=event(step); t=e['metadata']['temporal_target_ns']
        c.accept(e,t-1); c.activate(t)
        if step in (0,6):
            assert c.count==1
            np.testing.assert_allclose(c.goal,step/100)
        else:assert c.count==2
    assert sum(a.args[0]=='temporal_ensemble_reset' for a in log.call_args_list)==2


def test_single_pending_duplicate_and_late_messages():
    c=TemporalController(.01,Mock()); e=event(0); t=e['metadata']['temporal_target_ns']
    c.accept(e,t-1); c.accept(e,t-1)
    assert c.pending is not None
    c.activate(t)
    e1=event(1); t1=e1['metadata']['temporal_target_ns']
    c.accept(e1,t1)
    assert c.pending is None and c.count==1
    c.accept(event(2),t1+1)
    with pytest.raises(ValueError,match='overflow'):
        c.accept(event(3),t1+INTERVAL+1)


@pytest.mark.parametrize('bad',['shape','nan','clock','interval','future'])
def test_invalid_temporal_inputs_rejected(bad):
    c=TemporalController(.01,Mock()); e=event(0); t=e['metadata']['temporal_target_ns']
    if bad=='shape': e['value']=pa.array([[0]*16]*49)
    if bad=='nan': e['value']=pa.array([[float('nan')]*16]*50)
    if bad=='clock': e['metadata']['temporal_target_ns']+=1
    if bad=='interval': e['metadata']['interval']=40_000_000
    if bad=='future':t-=INTERVAL
    with pytest.raises(ValueError): c.accept(e,t-1)


def test_missing_slot_holds_last_filtered_position_and_zeros_reference_velocity():
    c=TemporalController(.01,Mock()); velocity=ReferenceVelocity()
    e=event(0); t=e['metadata']['temporal_target_ns']
    c.accept(e,t-1); c.sample(t)
    e=event(1); t=e['metadata']['temporal_target_ns']
    c.accept(e,t-1)
    for offset in range(0,INTERVAL,4_000_000):
        q,m=c.sample(t+offset)
        velocity.attach('move_position_right',q[:8],m)
    previous=q.copy()
    q,m=c.sample(t+INTERVAL)
    np.testing.assert_array_equal(q,previous)
    m=velocity.attach('move_position_right',q[:8],m)
    assert m['reference_end'] and json.loads(m['reference_velocity_json'])==[0]*8
    assert c.sample(t+2*INTERVAL) is None
    c.clear()
    assert c.pending is None and c.sample(t+3*INTERVAL) is None


def test_continuous_slots_do_not_reset_reference_every_step():
    c=TemporalController(.01,Mock())
    for step in range(4):
        e=event(step); t=e['metadata']['temporal_target_ns']
        c.accept(e,t-1)
        _,m=c.sample(t)
        assert m['reference_reset']==(step==0)
        assert not m['reference_end']


@pytest.mark.parametrize('bad',[None,True,-1000.,float('nan'),float('inf')])
def test_invalid_temporal_coefficients_fail_before_start(tmp_path,monkeypatch,bad):
    from deployment import chunk_execution
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/deployment.json').write_text(json.dumps(dict(action_execution=dict(
        mode='act_temporal',n_action_steps=1,temporal_ensemble_coeff=bad))))
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='coefficient'):chunk_execution.execution_settings()


@pytest.mark.parametrize('coefficient,latest_weight',[
    (-0.1047709150247644,.1),(-0.046041184561757126,.05),(0.02555339198176399,.01),
    (0.04743340266055522,.005),(0.06216167984441909,.003),(0.04192688834388196,.006)])
def test_configuration_assigns_requested_latest_weight(tmp_path,monkeypatch,coefficient,latest_weight):
    from deployment import chunk_execution
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/deployment.json').write_text(json.dumps(dict(action_execution=dict(
        mode='act_temporal',n_action_steps=1,temporal_ensemble_coeff=coefficient))))
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path)
    config=chunk_execution.execution_settings()
    coefficient=config['temporal_ensemble_coeff']
    ensemble=ACTTemporalEnsembler(coefficient,50)
    for _ in range(49):
        ensemble.update(torch.zeros((1,50,16)))
    actual=ensemble.update(torch.ones((1,50,16)))
    torch.testing.assert_close(actual,torch.full((1,16),latest_weight),rtol=1e-6,atol=1e-7)
    # A gap/reset has no older prediction to blend: use the first new target fully.
    ensemble.reset()
    torch.testing.assert_close(ensemble.update(torch.ones((1,50,16))),torch.ones((1,16)))


def test_switch_from_full50_does_not_silently_keep_chunk_execution(tmp_path,monkeypatch):
    from deployment import chunk_execution
    (tmp_path/'assets').mkdir()
    cfg=dict(mode='act_temporal',n_action_steps=50,temporal_ensemble_coeff=.01)
    path=tmp_path/'assets/deployment.json'
    path.write_text(json.dumps(dict(action_execution=cfg)))
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='n_action_steps=1'):chunk_execution.execution_settings()
    cfg['n_action_steps']=1; cfg['transition_ms']=150
    path.write_text(json.dumps(dict(action_execution=cfg)))
    with pytest.raises(ValueError,match='prefetch/transition'):chunk_execution.execution_settings()


@pytest.mark.parametrize('delay,stop_during',[(8,False),(80,False),(8,True)])
def test_client_clock_skip_no_burst_and_stop_during_inference(tmp_path,delay,stop_during):
    clock=[1_000_000_000]; queries=[]; sent=[]; stop=[False]
    observation=pa.array([{'id':1,'position':[0.]*16}])
    obs=dict(type='INPUT',id='observation',value=observation,
             metadata=dict(observation_monotonic_ns=clock[0],source_age_ns=0))
    queue=[obs]; log=Mock()
    class Node:
        def is_empty(self):
            if len(queries)>=5 and not queue:queue.append(dict(type='STOP'))
            return not queue
        def next(self):return queue.pop(0)
        def send_output(self,*args):sent.append(args)
    class IO:
        def write(self,s):queries.append(json.loads(s))
        def flush(self):pass
        def readline(self):
            clock[0]+=int((delay if len(queries)==2 else 8)*1e6)
            if stop_during:
                stop[0]=True; queue.append(dict(type='STOP'))
            return json.dumps(dict(interval=INTERVAL,positions=[[0.]*16]*50))
    def sleep(t):clock[0]+=round(t*1e9)
    run(Node(),IO(),tmp_path,log,now=lambda:clock[0],sleep=sleep,stopped=lambda:stop[0])
    assert not list(tmp_path.iterdir())
    if stop_during:
        assert not sent
        return
    steps=[r['metadata']['temporal_step'] for r in queries]
    assert steps==([0,1,3,4,5] if delay==80 else [0,1,2,3,4])
    assert len(sent)==(4 if delay==80 else 5)
    for _,_,m in sent:
        assert m['temporal_target_ns']==m['temporal_epoch_ns']+(m['temporal_step']+1)*INTERVAL


@pytest.mark.parametrize('name',['stop','disable'])
def test_real_executor_loop_flushes_pending_on_stop_without_emitting(tmp_path,monkeypatch,name):
    from deployment import temporal_execution as module
    monkeypatch.setattr(module,'session',lambda:tmp_path)
    clock=[1_000_000_000]; sent=[]; queue=[event(0)]; ticks=[0]
    class Node:
        def is_empty(self):return not queue
        def next(self):return queue.pop(0)
        def send_output(self,*args):sent.append(args)
    def sleep(t):
        ticks[0]+=1; clock[0]+=round(t*1e9)
        if ticks[0]==10:(tmp_path/(name+'.request')).touch()
        if ticks[0]==80:queue.append(dict(type='STOP'))
    module.run(Node(),dict(temporal_ensemble_coeff=.01),Mock(),now=lambda:clock[0],sleep=sleep)
    assert not sent
