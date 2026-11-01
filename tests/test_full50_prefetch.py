"""Full-segment prefetch changes request timing, never trajectory samples."""
import asyncio
import importlib.util
import json
from unittest.mock import Mock
import numpy as np
import pyarrow as pa
import pytest
from dora_openarm_actions_executor.main import BiquadLowpass
from deployment.chunk_executor import execute_chunk
from deployment.common import ROOT
from deployment.feedforward import ReferenceVelocity


def saved_event():
    saved=json.loads((ROOT/'tests/fixtures/release50_trial_phio1fa8.json').read_text())['prediction']
    return dict(type='INPUT',id='actions',value=pa.array(saved['positions'],type=pa.list_(pa.float32())),
        metadata=dict(interval=33_333_333,chunk_id='1',reset=True,observation_monotonic_ns=0))


@pytest.mark.parametrize('observation_age_ms',[80,400])
def test_all_samples_equal_archived_full50_even_when_inference_is_late(observation_age_ms):
    path=ROOT/'tests/reference/chunk_executor_full50.py'
    spec=importlib.util.spec_from_file_location('deployment._prefetch_baseline',path)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    def run(fn,prefetch):
        clock=[observation_age_ms*1_000_000];sent=[];rv=ReferenceVelocity()
        class Node:
            def send_output(self,name,value,metadata):
                if name.startswith('move_position'):
                    metadata=rv.attach(name,value.field('qpos')[0].as_py(),metadata)
                sent.append((clock[0],name,value,metadata))
        async def sleep(delay):clock[0]+=round(delay*1e9)
        async def go():
            filt=BiquadLowpass(250,15)
            for n in range(2):
                e=saved_event();e['metadata'].update(reset=n==0,chunk_id=str(n+1),
                    observation_monotonic_ns=clock[0]-observation_age_ms*1_000_000)
                kwargs=dict(now=lambda:clock[0],sleep=sleep)
                if prefetch:kwargs['prefetch_ms']=150
                assert await fn(Node(),e,50,filt,lambda:False,Mock(),**kwargs)
        asyncio.run(go())
        return sent
    old=run(baseline.execute_chunk,False);new=run(execute_chunk,True)
    a=[s for s in old if s[1]=='move_position_right']
    b=[s for s in new if s[1]=='move_position_right']
    assert len(a)==len(b)==834
    for before,after in zip(a,b):
        assert before[0]==after[0]
        assert before[2].equals(after[2])
        assert before[3]['reference_velocity_json']==after[3]['reference_velocity_json']
        assert after[3]['skipped_prefix_ns']==0
    assert b[416][3]['coarse_step']==49 and b[417][3]['coarse_step']==0
    assert b[-1][2].field('qpos')[0].as_py()[7]<-.45
    credits=[s for s in new if s[1]=='chunk_prefetch']
    done=[s for s in new if s[1]=='chunk_done']
    assert len(credits)==len(done)==2
    assert all(146_000_000<=d[0]-c[0]<=150_000_000 for c,d in zip(credits,done))


@pytest.mark.parametrize('stop_ns',[1_400_000_000,1_600_000_000,1_665_000_000])
def test_stop_before_or_after_prefetch_cancels_current_segment(stop_ns):
    clock=[0];sent=[]
    class Node:
        def send_output(self,*args):sent.append((clock[0],*args))
    async def sleep(delay):clock[0]+=round(delay*1e9)
    assert not asyncio.run(execute_chunk(Node(),saved_event(),50,BiquadLowpass(250,15),
        lambda:clock[0]>=stop_ns,Mock(),now=lambda:clock[0],sleep=sleep,prefetch_ms=150))
    assert all(s[0]<stop_ns for s in sent)
    assert not any(s[1]=='chunk_done' for s in sent)


@pytest.mark.parametrize('bad',[0,-1,501,True,150.0,None])
def test_prefetch_config_rejects_bad_timing(monkeypatch,tmp_path,bad):
    from deployment import chunk_execution
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/deployment.json').write_text(json.dumps({'action_execution':{
        'mode':'chunked_prefetch','n_action_steps':50,'prefetch_ms':bad,'temporal_ensemble_coeff':None}}))
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='prefetch_ms'):chunk_execution.execution_settings()
