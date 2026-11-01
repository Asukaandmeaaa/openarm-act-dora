"""Finite-block scheduling tests; fake clock/transport, never hardware."""
import asyncio
import io
import json
from unittest.mock import Mock
import numpy as np
import pyarrow as pa
import pytest
from deployment.chunk_execution import ChunkRequestGate, prepare_chunk
from deployment.chunk_executor import execute_chunk
from deployment.feedforward import ReferenceVelocity


def chunk():
    q = np.zeros((50, 16), dtype=np.float32)
    q[:7, :7] = np.arange(7)[:, None]*.001
    q[:7, 7] = np.linspace(-.08, -.40, 7)
    q[7:] = 10  # discarded tail must never enter interpolation
    return dict(type='INPUT', id='actions', value=pa.array(q.tolist(), type=pa.list_(pa.float32())),
                metadata=dict(interval=33_333_333, reset=True, chunk_id='a'))


def test_only_first_seven_targets_occupy_seven_slots():
    offsets, values, duration = prepare_chunk(chunk(), 7)
    assert duration == 7*33_333_333
    assert offsets[0] == 0 and offsets[-1] < duration
    assert np.all(np.diff(offsets) == 4_000_000)
    assert values.shape == (59, 16)
    assert np.max(values[:, :7]) <= .006001
    np.testing.assert_allclose(values[-1, :7], .006, atol=1e-7)
    assert values[-1, 7] == pytest.approx(-.40)
    assert np.all(values[offsets >= 200_000_000] == values[-1])


def test_completion_and_fresh_observation_required_not_a_timer():
    gate = ChunkRequestGate(True)
    def obs(t): return dict(id='observation', metadata={'observation_monotonic_ns':t})
    assert gate.accept(obs(1))
    gate.dispatched('a')
    assert not gate.accept(obs(10**12))  # time alone never unblocks execution
    with pytest.raises(RuntimeError): gate.dispatched('b')
    assert not gate.accept(dict(id='chunk_done', metadata=dict(chunk_id='wrong', completed_monotonic_ns=200)))
    assert not gate.accept(obs(201))
    assert not gate.accept(dict(id='chunk_done', metadata=dict(chunk_id='a', completed_monotonic_ns=200)))
    assert not gate.accept(obs(199))
    assert not gate.accept(obs(200))
    assert gate.accept(obs(201))
    gate.dispatched('b')
    gate.accept(dict(id='chunk_done', metadata=dict(chunk_id='a', completed_monotonic_ns=300)))
    assert not gate.accept(obs(400))  # duplicate ack cannot unblock another block


def test_legacy_mode_does_not_gate_observations():
    gate = ChunkRequestGate(False)
    gate.dispatched('a')
    assert gate.accept(dict(id='observation', metadata={}))
    assert not gate.accept(dict(id='chunk_done', metadata={}))


@pytest.mark.parametrize('stop_ns', [None, 80_000_000, 233_000_000])
def test_scheduler_completion_and_interrupt_with_fake_time(stop_ns):
    from dora_openarm_actions_executor.main import BiquadLowpass
    current = [0]; sent = []; rv = ReferenceVelocity()
    class Node:
        def send_output(self, name, value, metadata):
            if name.startswith('move_position'):
                metadata = rv.attach(name, value.field('qpos')[0].as_py(), metadata)
            sent.append((current[0], name, value, metadata))
    async def sleep(delay): current[0] += round(delay*1e9)
    stopped = lambda: stop_ns is not None and current[0] >= stop_ns
    result = asyncio.run(execute_chunk(Node(), chunk(), 7, BiquadLowpass(250, 15),
                         stopped, Mock(), now=lambda:current[0], sleep=sleep))
    moves = [s for s in sent if s[1] == 'move_position_right']
    if stop_ns is None:
        assert result
        assert len(moves) == 59
        assert set(s[3]['coarse_step'] for s in moves) == set(range(7))
        assert sent[-1][1] == 'chunk_done'
        assert sent[-1][0] == 233_333_331
        assert all(v == 0 for v in json.loads(moves[-1][3]['reference_velocity_json']))
    else:
        assert not result
        assert all(s[0] < stop_ns for s in moves)
        assert not any(s[1] == 'chunk_done' for s in sent)
    assert all(v == 0 for v in json.loads(moves[0][3]['reference_velocity_json']))




@pytest.mark.parametrize('mutation', ['nan', 'short', 'interval'])
def test_invalid_chunks_rejected_before_output(mutation):
    event = chunk()
    if mutation == 'nan':
        q = np.zeros((50,16)); q[0,0] = np.nan
        event['value'] = pa.array(q.tolist())
    elif mutation == 'short': event['value'] = event['value'][:6]
    else: event['metadata']['interval'] = 1
    with pytest.raises(ValueError): prepare_chunk(event, 7)


@pytest.mark.parametrize('stop_ns', [None, 800_000_000, 1_665_000_000])
def test_full_fifty_reaches_late_opening_or_interrupts_before_completion(stop_ns):
    from dora_openarm_actions_executor.main import BiquadLowpass
    q=np.zeros((50,16),dtype=np.float32)
    q[:,0]=np.linspace(0,.3,50)
    q[:,7]=np.r_[np.full(25,-.1),np.linspace(-.1,-.45,25)]
    event=chunk();event['value']=pa.array(q.tolist(),type=pa.list_(pa.float32()))
    current=[0];sent=[];rv=ReferenceVelocity()
    class Node:
        def send_output(self,name,value,metadata):
            if name.startswith('move_position'):
                metadata=rv.attach(name,value.field('qpos')[0].as_py(),metadata)
            sent.append((current[0],name,value,metadata))
    async def sleep(delay):current[0]+=round(delay*1e9)
    result=asyncio.run(execute_chunk(Node(),event,50,BiquadLowpass(250,15),
        lambda:stop_ns is not None and current[0]>=stop_ns,Mock(),
        now=lambda:current[0],sleep=sleep))
    right=[s for s in sent if s[1]=='move_position_right']
    if stop_ns is None:
        assert result and len(right)==417
        assert {s[3]['coarse_step'] for s in right}==set(range(50))
        assert right[-1][2].field('qpos')[0].as_py()[7]<-.44
        assert right[-1][2].field('qpos')[0].as_py()[0]>.29
        assert sent[-1][1]=='chunk_done' and sent[-1][0]==1_666_666_650
        assert json.loads(right[-1][3]['reference_velocity_json'])==[0]*8
    else:
        assert not result and not any(s[1]=='chunk_done' for s in sent)
        assert all(s[0]<stop_ns for s in right)
    assert json.loads(right[0][3]['reference_velocity_json'])==[0]*8


def test_client_waits_for_matching_completion_and_fresh_sources(monkeypatch,tmp_path):
    from dora_openarm_local_policy_server import main as client
    def obs(i,t,age=0):
        return dict(type='INPUT',id='observation',value=pa.array([{'id':i,'position':[0.]*16}]),
                    metadata=dict(observation_monotonic_ns=t,source_age_ns=age))
    def done(cid,t):
        return dict(type='INPUT',id='chunk_done',metadata=dict(chunk_id=cid,completed_monotonic_ns=t))
    events=[obs(1,10),obs(2,20),done('wrong',100),obs(3,150),done('10',200),
            obs(4,201,5),obs(5,205,1),done('10',300),obs(6,350),done('205',400),obs(7,420,10)]
    class Node:
        send_output=Mock()
        def __iter__(self):return iter(events)
    class Transport(io.StringIO):
        def readline(self):return json.dumps(dict(interval=33_333_333,positions=np.zeros((50,16)).tolist()))
    node=Node();transport=Transport()
    monkeypatch.setattr(client.dora,'Node',lambda:node)
    monkeypatch.setattr(client,'Log',lambda _:Mock())
    monkeypatch.setattr(client,'execution_settings',lambda:{'mode':'chunked','n_action_steps':50})
    client._main_dora(transport,tmp_path)
    requests=[json.loads(s) for s in transport.getvalue().splitlines()]
    assert [r['metadata']['observation_monotonic_ns'] for r in requests]==[10,205,420]
    assert [r['reset'] for r in requests]==[True,False,False]
    assert [c.args[2]['chunk_id'] for c in node.send_output.call_args_list]==['10','205','420']
    assert all(len(c.args[1])==50 for c in node.send_output.call_args_list)


@pytest.mark.parametrize('bad', [0,10,49,51,True,50.0])
def test_runtime_requires_explicit_complete_fifty(monkeypatch,tmp_path,bad):
    from deployment import chunk_execution
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/deployment.json').write_text(json.dumps({'action_execution':{
        'mode':'chunked','n_action_steps':bad,'temporal_ensemble_coeff':None}}))
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='n_action_steps=50'):chunk_execution.execution_settings()
