"""Two-chunk alignment and real client wiring with fake transport only."""
import io
import json
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest

from deployment.tail_blend import PreviousTailBlender


def trajectory(offset):
    return np.repeat((np.arange(100, dtype=np.float32) + offset)[:, None], 16, axis=1)


def test_uses_unexecuted_raw_tail_without_recursive_history_or_input_mutation():
    blender = PreviousTailBlender(.01)
    first, second, third = [trajectory(x) for x in (0, 1000, 2000)]
    a, info = blender.apply(first, chunk_id='a', reset=True)
    np.testing.assert_array_equal(a, first)
    assert not info['applied'] and info['effective_new_weight'] == 1
    b, info = blender.apply(second, chunk_id='b')
    np.testing.assert_allclose(b[:50], .99*first[50:] + .01*second[:50])
    np.testing.assert_array_equal(b[50:], second[50:])
    assert info['previous_chunk_id'] == 'a'
    assert info['effective_old_weight'] == .99
    c, info = blender.apply(third, chunk_id='c')
    np.testing.assert_allclose(c[:50], .99*second[50:] + .01*third[:50])
    np.testing.assert_array_equal(second, trajectory(1000))
    assert info['previous_chunk_id'] == 'b'


@pytest.mark.parametrize('reset', [True, False])
def test_episode_reset_or_clear_discards_previous_prediction(reset):
    blender = PreviousTailBlender(.01)
    blender.apply(trajectory(0), chunk_id='a')
    if not reset:
        blender.clear()
    result, info = blender.apply(trajectory(1000), chunk_id='b', reset=reset)
    np.testing.assert_array_equal(result, trajectory(1000))
    assert not info['applied']


def test_disabled_preserves_old_behavior_and_does_not_cache():
    blender = PreviousTailBlender()
    for i in range(3):
        q = trajectory(i)
        result, info = blender.apply(q, chunk_id=str(i))
        np.testing.assert_array_equal(result, q)
        assert not info['enabled'] and blender.tail is None


@pytest.mark.parametrize('bad', [True, -.01, 1.01, float('nan'), float('inf'), '0.01'])
def test_invalid_weights_rejected(bad):
    with pytest.raises(ValueError):
        PreviousTailBlender(bad)


@pytest.mark.parametrize('bad', [np.zeros((50,16)), np.full((100,16), np.nan)])
def test_invalid_predictions_do_not_replace_history(bad):
    blender = PreviousTailBlender(.01)
    blender.apply(trajectory(0), chunk_id='a')
    with pytest.raises(ValueError):
        blender.apply(bad, chunk_id='bad')
    assert blender.chunk_id == 'a'


@pytest.mark.parametrize('weight', [.01, None, True, 2])
def test_runtime_config_validates_explicit_weight(monkeypatch, tmp_path, weight):
    from deployment import chunk_execution
    monkeypatch.delenv('OPENARM_DEPLOYMENT_CONFIG', raising=False)
    monkeypatch.setattr(chunk_execution, 'ROOT', tmp_path)
    (tmp_path/'assets').mkdir()
    (tmp_path/'assets/deployment.json').write_text(json.dumps({'action_execution': {
        'mode':'chunk100_exec50', 'chunk_size':100, 'n_action_steps':50,
        'tail_blend_new_weight':weight}}))
    if weight == .01:
        assert chunk_execution.execution_settings()['tail_blend_new_weight'] == .01
    else:
        with pytest.raises(ValueError):
            chunk_execution.execution_settings()


@pytest.mark.parametrize('stop_in_second_inference', [False, True])
def test_real_client_logs_raw_and_sends_blended_then_resets_or_stops(monkeypatch, tmp_path, stop_in_second_inference):
    from deployment import fifty_client as client
    def observation(i, t):
        return dict(type='INPUT', id='observation', value=pa.array([{'id':i,'position':[0.]*16}]),
                    metadata=dict(observation_monotonic_ns=t, source_age_ns=0))
    def done(cid, t):
        return dict(type='INPUT', id='chunk_done',
                    metadata=dict(chunk_id=cid, completed_monotonic_ns=t))
    events = [observation(1,10), observation(2,15), done('10',20), observation(2,30),
              done('30',40), observation(0,50)]
    node = Mock()
    node.__iter__ = Mock(return_value=iter(events))
    predictions = [trajectory(x) for x in (0,1000,2000)]
    class Transport(io.StringIO):
        count = 0
        def readline(self):
            q = predictions[self.count]; self.count += 1
            if stop_in_second_inference and self.count == 2:
                (tmp_path/'stop.request').touch()
            return json.dumps(dict(interval=33333333, cutoff_hz=15, positions=q.tolist()))
    transport = Transport(); fake_socket = Mock()
    fake_socket.makefile.return_value = transport
    monkeypatch.setattr(client.socket, 'socket', Mock(return_value=fake_socket))
    monkeypatch.setenv('SOCKET', 'unused_fake_socket')
    monkeypatch.setattr(client.dora, 'Node', lambda:node)
    monkeypatch.setattr(client, 'session', lambda:tmp_path)
    monkeypatch.setattr(client, 'ObservationSnapshots', Mock())
    logs = []
    monkeypatch.setattr(client, 'Log', lambda role: lambda event, **kw: logs.append(dict(event=event, **kw)))
    monkeypatch.setattr(client, 'execution_settings', lambda: {
        'mode':'chunk100_exec50','chunk_size':100,'n_action_steps':50,'tail_blend_new_weight':.01})
    shared = tmp_path/'shared'; shared.mkdir()
    monkeypatch.setattr(client.tempfile, 'mkdtemp', lambda **kw:str(shared))
    blender = PreviousTailBlender(.01)
    monkeypatch.setattr(client, 'PreviousTailBlender', lambda weight:blender)
    client.main()
    calls = node.send_output.call_args_list
    assert len(calls) == (1 if stop_in_second_inference else 3)
    np.testing.assert_array_equal(calls[0].args[1].to_pylist(), predictions[0])
    if not stop_in_second_inference:
        expected = .99*predictions[0][50:] + .01*predictions[1][:50]
        np.testing.assert_allclose(calls[1].args[1].to_pylist()[:50], expected)
        raw = [x for x in logs if x['event']=='raw_policy_chunk'][1]
        np.testing.assert_array_equal(raw['positions'], predictions[1])
        np.testing.assert_allclose(raw['committed'], expected)
        assert calls[1].args[2]['tail_blend_new_weight'] == .01
        np.testing.assert_array_equal(calls[2].args[1].to_pylist(), predictions[2])
        assert calls[2].args[2]['tail_blend_new_weight'] == 1.0
    else:
        assert any(x['event']=='prediction_discarded_after_stop' for x in logs)
    assert blender.tail is None
