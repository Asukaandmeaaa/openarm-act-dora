"""Pipeline scheduling with fake clocks/transport only. No devices."""
import asyncio
import io
import json
from unittest.mock import Mock
import numpy as np
import pyarrow as pa
import pytest
from dora_openarm_actions_executor.main import BiquadLowpass
from deployment.pipeline_execution import PipelineRequestGate, aligned_chunk, execute_pipeline_chunk


def event(observed=0, cid='1'):
    q = np.zeros((50, 16))
    q[:, 0] = np.linspace(.1, .3, 50)
    q[:, 7] = np.r_[np.full(30, -.1), np.linspace(-.1, -.46, 20)]
    return dict(type='INPUT', id='actions', value=pa.array(q.tolist(), type=pa.list_(pa.float32())),
                metadata=dict(interval=33_333_333, chunk_id=cid, observation_monotonic_ns=observed))


def obs(t, age=0):
    return dict(type='INPUT', id='observation', metadata=dict(observation_monotonic_ns=t, source_age_ns=age))


def credit(cid, t):
    return dict(type='INPUT', id='chunk_prefetch', metadata=dict(chunk_id=cid, requested_monotonic_ns=t))


def test_one_credit_ignores_wrong_duplicate_completion_and_stale_sources():
    gate = PipelineRequestGate()
    assert gate.accept(obs(10))
    gate.dispatched('10')
    assert not gate.accept(obs(10000))
    gate.accept(credit('wrong', 20))
    assert not gate.accept(obs(30))
    gate.accept(credit('10', 100))
    assert not gate.accept(obs(110, 20))
    assert gate.accept(obs(120, 10))
    gate.dispatched('120')
    gate.accept(credit('10', 130))
    gate.accept(dict(id='chunk_done', metadata=dict(chunk_id='120', completed_monotonic_ns=140)))
    assert not gate.accept(obs(150))
    with pytest.raises(RuntimeError):
        gate.dispatched('third')


@pytest.mark.parametrize('transition', [0, 100, 150])
def test_alignment_keeps_tail_and_does_not_append_transition(transition):
    first = aligned_chunk(event(), 100_000_000, None, transition)
    assert first[2] == 50*33_333_333 and first[3] == 0
    p = np.zeros(16)
    plan = aligned_chunk(event(), 150_000_000, p, transition)
    offsets, values, duration, age, _ = plan
    assert age == 150_000_000 and age//33_333_333 == 4
    assert duration == 50*33_333_333-age
    assert values[-1, 7] == pytest.approx(-.46)
    assert values[-1, 0] == pytest.approx(.3)
    if transition:
        np.testing.assert_array_equal(values[0], p)
        raw = aligned_chunk(event(), 150_000_000, p, 0)[1]
        np.testing.assert_allclose(values[offsets >= transition*1e6], raw[offsets >= transition*1e6])


def test_expired_prediction_requests_fresh_one_without_movement():
    sent = []
    class Node:
        def send_output(self, *args): sent.append(args)
    p = np.zeros(16)
    result, complete = asyncio.run(execute_pipeline_chunk(Node(), event(),
        dict(prefetch_ms=150, transition_ms=100), BiquadLowpass(250,15), p,
        lambda:False, Mock(), now=lambda:2_000_000_000))
    assert complete and result is p
    assert [s[0] for s in sent] == ['chunk_prefetch']


def test_late_result_uses_elapsed_time_not_a_fixed_four_step_skip():
    p=np.zeros(16)
    ontime=aligned_chunk(event(),100_000_000,p,150)
    late=aligned_chunk(event(),350_000_000,p,150)
    assert late[3]//33_333_333==10
    assert ontime[2]-late[2]==250_000_000
    assert late[1][-1,7]==pytest.approx(-.46)
    assert aligned_chunk(event(),1_600_000_000,p,150) is None
    with pytest.raises(ValueError,match='future'):
        aligned_chunk(event(2_000_000_000),1_000_000_000,p,150)


@pytest.mark.parametrize('stop_time', [None, 50_000_000, 1_450_000_000, 1_565_000_000])
def test_prefetch_does_not_cut_tail_and_stop_interrupts(stop_time):
    clock = [100_000_000]; sent = []; log = Mock()
    class Node:
        def send_output(self, name, value, metadata): sent.append((clock[0],name,value,metadata))
    async def sleep(delay): clock[0] += round(delay*1e9)
    stop = lambda:stop_time is not None and clock[0] >= 100_000_000+stop_time
    _, complete = asyncio.run(execute_pipeline_chunk(Node(), event(),
        dict(prefetch_ms=150, transition_ms=100), BiquadLowpass(250,15), np.zeros(16),
        stop, log, now=lambda:clock[0], sleep=sleep))
    moves = [s for s in sent if s[1]=='move_position_right']
    prefetch = [s for s in sent if s[1]=='chunk_prefetch']
    if stop_time is None:
        assert complete and len(prefetch)==1
        assert 146_000_000 <= clock[0]-prefetch[0][0] <= 150_000_000
        assert moves[-1][2].field('qpos')[0].as_py()[7] < -.45
        assert moves[-1][3]['coarse_step']==49
        assert sent[-1][1]=='chunk_done'
    else:
        assert not complete
        assert all(s[0]<100_000_000+stop_time for s in moves)
        assert not any(s[1]=='chunk_done' for s in sent)


@pytest.mark.parametrize('stop_during_inference', [False, True])
@pytest.mark.parametrize('mode', ['pipelined', 'chunked_prefetch'])
def test_real_client_single_prefetch_and_stop_discards_response(monkeypatch,tmp_path,stop_during_inference,mode):
    from dora_openarm_local_policy_server import main as client
    events=[obs(10),obs(20),credit('10',100),obs(110,20),obs(120,10),
            credit('10',130),obs(140),credit('120',200),obs(220)]
    for i,e in enumerate(events):
        if e['id']=='observation':e['value']=pa.array([{'id':i,'position':[0.]*16}])
    class Node:
        send_output=Mock()
        def __iter__(self):return iter(events)
    class Transport(io.StringIO):
        def readline(self):
            if stop_during_inference:(tmp_path/'stop.request').touch()
            return json.dumps(dict(interval=33_333_333,positions=np.zeros((50,16)).tolist()))
    node=Node();transport=Transport()
    monkeypatch.setattr(client.dora,'Node',lambda:node)
    monkeypatch.setattr(client,'Log',lambda _:Mock())
    monkeypatch.setattr(client,'session',lambda:tmp_path)
    monkeypatch.setattr(client,'execution_settings',lambda:{'mode':mode})
    client._main_dora(transport,tmp_path)
    ids=[c.args[2]['chunk_id'] for c in node.send_output.call_args_list]
    assert ids == ([] if stop_during_inference else ['10','120','220'])


@pytest.mark.parametrize('mode', ['pipelined', 'chunked_prefetch'])
def test_queued_chunk_cannot_execute_after_stop(monkeypatch,tmp_path,mode):
    from deployment import pipeline_execution, chunk_executor
    module = pipeline_execution if mode=='pipelined' else chunk_executor
    e=event(cid='1');e['metadata']['observation_monotonic_ns']=0
    sent=[];calls=[0]
    class Node:
        def is_empty(self):return False
        def next(self):
            calls[0]+=1
            if calls[0]==1:return e
            if calls[0]==2:
                (tmp_path/'stop.request').touch()
                return event(cid='2')  # prediction already buffered when STOP arrives
            return None
        def send_output(self,*args):sent.append(args)
    async def fake_execute(node,event,*args,**kwargs):
        node.send_output('move_position_right',None,{'chunk_id':event['metadata']['chunk_id']})
        return np.zeros(16),True
    monkeypatch.setattr(module,'session',lambda:tmp_path)
    monkeypatch.setattr(module,'execute_pipeline_chunk' if mode=='pipelined' else 'execute_chunk',fake_execute)
    if mode=='pipelined':asyncio.run(module.run(Node(),{},Mock()))
    else:asyncio.run(module.run(Node(),50,Mock(),prefetch_ms=150))
    assert [s[2]['chunk_id'] for s in sent]==['1']
