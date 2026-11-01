"""Official dataflow contracts and passive recording; no devices or dataflow."""
import io
import json
import threading
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
import yaml

from deployment.common import ROOT
from deployment import observation_recording as recording


def test_chunk_graphs_connect_completion_and_keep_action_queue():
    reference=ROOT/'tests/reference/dataflow-cell-sample.yaml'
    official={n['id']:n for n in yaml.safe_load(reference.read_text())['nodes']}
    assert official['actions-executor']['inputs']['actions']=='policy-server/actions'
    for name in ('dataflow.yml','dataflow-sim.yml'):
        nodes={n['id']:n for n in yaml.safe_load((ROOT/name).read_text())['nodes']}
        assert nodes['executor']['inputs']=={'actions':{'source':'policy-client/actions','queue_size':1}}
        assert 'chunk_done' in nodes['executor']['outputs']
        assert nodes['policy-client']['inputs']=={'observation':{'source':'observer/observation','queue_size':1},'chunk_done':'executor/chunk_done','chunk_prefetch':'executor/chunk_prefetch'}
        assert 'chunk_prefetch' in nodes['executor']['outputs']


def test_client_processes_every_observation_without_completion_ack(monkeypatch,tmp_path):
    from dora_openarm_local_policy_server import main as client
    monkeypatch.setattr(client,'execution_settings',lambda:{'mode':'official_blend'})
    events=[dict(type='INPUT',id='observation',value=pa.array([{'id':n,'position':[0.]*16}]),
                 metadata={'observation_monotonic_ns':n}) for n in (1,2,3)]
    class Node:
        send_output=Mock()
        def __iter__(self):return iter(events)
    class Transport(io.StringIO):
        def readline(self):return json.dumps({'interval':33_333_333,'positions':np.zeros((50,16)).tolist()})
    node=Node();transport=Transport()
    monkeypatch.setattr(client.dora,'Node',lambda:node)
    monkeypatch.setattr(client,'Log',lambda _:Mock())
    monkeypatch.setattr(client,'session',lambda:tmp_path)
    client._main_dora(transport,tmp_path)
    requests=[json.loads(line) for line in transport.getvalue().splitlines()]
    assert [r['metadata']['observation_monotonic_ns'] for r in requests]==[1,2,3]
    assert [r['reset'] for r in requests]==[True,False,False]
    assert all(len(call.args[1])==50 for call in node.send_output.call_args_list)
    # Every request is attempted; a bounded recorder may explicitly report busy.
    records=[json.loads(line) for line in (tmp_path/'observations/index.jsonl').read_text().splitlines()]
    assert sorted(r['observation_id'] for r in records)==[1,2,3]
    for row in records:
        if row['event']=='observation_saved':
            with pa.memory_map(str(tmp_path/'observations'/row['file']),'r') as source:
                batch=pa.ipc.open_file(source).read_all()
                assert batch['id'][0].as_py()==row['observation_id']


def test_snapshot_survives_source_unlink_and_preserves_bytes(monkeypatch,tmp_path):
    source=tmp_path/'input.arrow'
    batch=pa.record_batch({'id':[42], 'position':[[.1]*16], 'camera_ceiling':[[1,2,3]]})
    with pa.OSFile(str(source),'wb') as f:
        with pa.ipc.new_file(f,batch.schema) as writer:writer.write(batch)
    expected=source.read_bytes()
    begun=threading.Event();release=threading.Event()
    original=recording.shutil.copyfileobj
    def delayed(stream,target):
        begun.set();assert release.wait(2);original(stream,target)
    monkeypatch.setattr(recording.shutil,'copyfileobj',delayed)
    snapshots=recording.ObservationSnapshots(tmp_path)
    try:
        snapshots.capture(source,42,{'observation_monotonic_ns':100})
        assert begun.wait(2)
        source.unlink()
    finally:
        release.set();snapshots.close()
    assert (tmp_path/'observations/0000.arrow').read_bytes()==expected
    index=json.loads((tmp_path/'observations/index.jsonl').read_text())
    assert index['observation_id']==42 and index['metadata']['observation_monotonic_ns']==100


def test_recording_failures_do_not_block_inference(monkeypatch,tmp_path):
    source=tmp_path/'input.arrow';source.write_bytes(b'test')
    monkeypatch.setattr(recording.shutil,'copyfileobj',Mock(side_effect=OSError('disk full')))
    snapshots=recording.ObservationSnapshots(tmp_path)
    snapshots.capture(source,1,{})
    snapshots.close()
    index=json.loads((tmp_path/'observations/index.jsonl').read_text())
    assert index['event']=='save_failed'
    assert not list((tmp_path/'observations').glob('*.partial'))
    assert not list((tmp_path/'observations').glob('*.arrow'))


def test_snapshot_rate_and_count_bound(monkeypatch,tmp_path):
    source=tmp_path/'input.arrow';source.write_bytes(b'test')
    now=[0];monkeypatch.setattr(recording.time,'monotonic_ns',lambda:now[0])
    snapshots=recording.ObservationSnapshots(tmp_path,max_samples=2)
    try:
        snapshots.capture(source,1,{})
        snapshots.pending.join()
        now[0]=500_000_000;snapshots.capture(source,2,{})
        now[0]=1_000_000_000;snapshots.capture(source,3,{})
        snapshots.pending.join()
        now[0]=2_000_000_000;snapshots.capture(source,4,{})
    finally:snapshots.close()
    index=[json.loads(l) for l in (tmp_path/'observations/index.jsonl').read_text().splitlines()]
    assert [r['observation_id'] for r in index]==[1,3]


def test_old_seven_step_config_cannot_silently_change_current_experiment(monkeypatch,tmp_path):
    from deployment import chunk_execution
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/deployment.json').write_text(json.dumps({'action_execution':{'mode':'chunked','n_action_steps':7}}))
    monkeypatch.setattr(chunk_execution,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='n_action_steps=50'):chunk_execution.execution_settings()
