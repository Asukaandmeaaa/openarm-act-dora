"""100-step official temporal ensemble, locked 50-target blocks, fake I/O."""
import json
from unittest.mock import Mock

import numpy as np
import pytest

from deployment.temporal_commit import CommittedTemporalController
from deployment.temporal_execution import TemporalController, make_controller
from deployment.temporal_client import INTERVAL
from test_temporal_execution import event


def test_locked50_matches_independent_same_time_sum_and_actual_weights():
    history = np.random.default_rng(71).normal(size=(220,100,16)).astype(np.float32)
    log = Mock()
    controller = CommittedTemporalController(0., log, chunk_size=100, steps=50)
    for step, q in enumerate(history):
        e = event(step, q); target = e['metadata']['temporal_target_ns']
        controller.accept(e, target-1)
        _, metadata = controller.sample(target)
        committed = step//50*50
        sources = range(max(0,step-99), committed+1)
        expected = np.mean([history[i,step-i] for i in sources],axis=0)
        np.testing.assert_allclose(controller.goal,expected,atol=4e-7,rtol=3e-5)
        assert metadata['n_action_steps'] == 50 and metadata['chunk_size'] == 100
        assert metadata['committed_source_step'] == committed
        assert metadata['committed_step_index'] == step-committed
        assert controller.counts[step-committed] == len(sources)
        assert controller.latest_weights[step-committed] == pytest.approx(1/len(sources))
        if step == 100:
            assert controller.latest_weights[0] == pytest.approx(.01)
            assert controller.latest_weights[49] == pytest.approx(1/51)
    assert len([c for c in log.call_args_list if c.args[0]=='temporal_block_committed']) == 5


def test_continuing_predictions_do_not_overwrite_fifty_locked_targets():
    c = CommittedTemporalController(0.,Mock(),chunk_size=100,steps=50)
    first = np.repeat(np.arange(100,dtype=np.float32)[:,None],16,axis=1)
    for step in range(51):
        e = event(step,first if step==0 else np.full((100,16),1000,dtype=np.float32))
        t = e['metadata']['temporal_target_ns']; c.accept(e,t-1); c.sample(t)
        if step < 50:
            np.testing.assert_array_equal(c.goal,step)
        else:
            assert np.all(c.goal > 900)


def test_missing_predictions_complete_fifty_then_hold_reset_and_recover():
    c = CommittedTemporalController(0.,Mock(),chunk_size=100,steps=50)
    q = np.repeat((np.arange(100,dtype=np.float32)/100)[:,None],16,axis=1)
    e = event(0,q); t = e['metadata']['temporal_target_ns']; c.accept(e,t-1)
    for i in range(50):
        previous,_ = c.sample(t+i*INTERVAL)
        np.testing.assert_array_equal(c.goal,q[i])
    held,m = c.sample(t+50*INTERVAL)
    np.testing.assert_array_equal(held,previous); assert m['reference_end']
    assert c.sample(t+51*INTERVAL) is None
    e = event(53,q); t = e['metadata']['temporal_target_ns']; c.accept(e,t-1)
    _,m = c.sample(t)
    assert m['reference_reset'] and c.fusion.count == 1


@pytest.mark.parametrize('request_name', ['stop','disable'])
def test_stop_clears_actual_fifty_step_runner_without_waiting_for_block(tmp_path,monkeypatch,request_name):
    from deployment import temporal_execution as module
    monkeypatch.setattr(module,'session',lambda:tmp_path)
    clock=[1_000_000_000];sent=[];queue=[event(0,np.zeros((100,16),np.float32))];stop_at=[]
    class Node:
        def is_empty(self):return not queue
        def next(self):return queue.pop(0)
        def send_output(self,*args):sent.append((clock[0],args))
    def sleep(t):
        clock[0]+=round(t*1e9)
        if clock[0]>=1_250_000_000 and not stop_at:
            (tmp_path/(request_name+'.request')).touch();stop_at.append(clock[0])
        if clock[0]>=1_500_000_000:queue.append(dict(type='STOP'))
    c=CommittedTemporalController(0.,Mock(),chunk_size=100,steps=50)
    module.run(Node(),dict(temporal_ensemble_coeff=0.,n_action_steps=50,chunk_size=100),Mock(),
               now=lambda:clock[0],sleep=sleep,controller=c)
    assert sent and all(t<stop_at[0] for t,_ in sent)
    assert c.block is None and c.fusion.pending is None and c.fusion.ensemble.ensembled_actions is None


def test_hundred_target_factory_and_explicit_config(monkeypatch,tmp_path):
    from deployment import chunk_execution
    monkeypatch.delenv('OPENARM_DEPLOYMENT_CONFIG',raising=False)
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path);(tmp_path/'assets').mkdir()
    cfg=dict(mode='act_temporal',chunk_size=100,n_action_steps=50,temporal_ensemble_coeff=0.)
    (tmp_path/'assets/deployment.json').write_text(json.dumps({'action_execution':cfg}))
    c=make_controller(chunk_execution.execution_settings(),Mock())
    assert c.steps == 50 and c.fusion.chunk_size == 100
    for change in [dict(chunk_size=True),dict(chunk_size=99),dict(n_action_steps=10),dict(tail_blend_new_weight=.01)]:
        (tmp_path/'assets/deployment.json').write_text(json.dumps({'action_execution':{**cfg,**change}}))
        with pytest.raises(ValueError):chunk_execution.execution_settings()


def test_gap_cannot_mix_misaligned_hundred_step_predictions():
    c=TemporalController(0.,Mock(),chunk_size=100)
    for step in (0,1,7):
        q=np.full((100,16),step,dtype=np.float32);e=event(step,q);t=e['metadata']['temporal_target_ns']
        c.accept(e,t-1);c.activate(t)
    assert c.count==1
    np.testing.assert_array_equal(c.goal,7)
