"""Committed ten-step targets, with synthetic predictions and clocks only."""
import json
from unittest.mock import Mock
import numpy as np
import pytest
from deployment.temporal_commit import CommittedTemporalController
from deployment.temporal_client import INTERVAL
from deployment.feedforward import ReferenceVelocity
from test_temporal_execution import event


def test_each_block_matches_independent_time_aligned_sum_at_commit_time():
    coefficient = .04743340266055522
    rng = np.random.default_rng(28)
    history = rng.normal(size=(120, 50, 16)).astype(np.float32)
    history[:, :, 8:] = 0
    c = CommittedTemporalController(coefficient, Mock())
    for step, q in enumerate(history):
        e = event(step, q); t = e['metadata']['temporal_target_ns']
        c.accept(e, t-1)
        _, metadata = c.sample(t)
        start = step//10*10
        contributors = range(max(0, step-49), start+1)
        aligned = np.array([history[i, step-i] for i in contributors])
        weights = np.exp(-coefficient*np.arange(len(aligned)))
        expected = np.average(aligned, axis=0, weights=weights)
        np.testing.assert_allclose(c.goal, expected, atol=5e-7, rtol=3e-5)
        assert metadata['n_action_steps'] == 10
        assert metadata['committed_source_step'] == start
        assert metadata['committed_step_index'] == step-start
        assert metadata['reference_reset'] == (step == 0)
        assert not metadata['reference_end']


def test_new_predictions_cannot_rewrite_active_block_but_affect_next_block():
    c = CommittedTemporalController(.04743340266055522, Mock())
    first = np.repeat(np.arange(50, dtype=np.float32)[:, None], 16, axis=1)
    for step in range(11):
        e = event(step, first if step == 0 else np.full((50, 16), 100.))
        t = e['metadata']['temporal_target_ns']
        c.accept(e, t-1); c.activate(t)
        if step < 10:
            np.testing.assert_array_equal(c.goal, step)
        else:
            assert np.all(c.goal > 10)


def test_missing_predictions_finish_locked_block_then_hold_and_recover():
    c = CommittedTemporalController(.01, Mock())
    q = np.repeat(np.arange(50, dtype=np.float32)[:, None]/100, 16, axis=1)
    e = event(0, q); t = e['metadata']['temporal_target_ns']
    c.accept(e, t-1)
    velocity = ReferenceVelocity()
    for index in range(10):
        position, m = c.sample(t+index*INTERVAL)
        velocity.attach('move_position_right', position[:8], m)
        np.testing.assert_array_equal(c.goal, q[index])
    previous = position.copy()
    position, m = c.sample(t+10*INTERVAL)
    np.testing.assert_array_equal(position, previous)
    m = velocity.attach('move_position_right', position[:8], m)
    assert m['reference_end'] and json.loads(m['reference_velocity_json']) == [0]*8
    assert c.sample(t+11*INTERVAL) is None
    e = event(12); t = e['metadata']['temporal_target_ns']
    c.accept(e, t-1); _, m = c.sample(t)
    assert m['reference_reset'] and m['committed_source_step'] == 12
    assert c.fusion.count == 1


def test_missed_execution_slots_abort_without_catchup_and_clear_discards_block():
    log = Mock(); c = CommittedTemporalController(.01, log)
    e = event(0); t = e['metadata']['temporal_target_ns']
    c.accept(e, t-1); previous, _ = c.sample(t)
    position, m = c.sample(t+3*INTERVAL)
    np.testing.assert_array_equal(previous, position)
    assert m['reference_end'] and c.block is None
    assert any(call.args[0] == 'temporal_block_aborted_late' for call in log.call_args_list)
    c.clear()
    assert c.sample(t+4*INTERVAL) is None
    assert c.fusion.pending is None


@pytest.mark.parametrize('name', ['stop', 'disable'])
def test_stop_mid_block_flushes_ten_targets_in_actual_loop(tmp_path, monkeypatch, name):
    from deployment import temporal_execution as module
    monkeypatch.setattr(module, 'session', lambda: tmp_path)
    clock = [1_000_000_000]; sent = []; queue = [event(0)]; stopped_at = []
    class Node:
        def is_empty(self): return not queue
        def next(self): return queue.pop(0)
        def send_output(self, *args): sent.append((clock[0], args))
    def sleep(t):
        clock[0] += round(t*1e9)
        if clock[0] >= 1_100_000_000 and not stopped_at:
            (tmp_path/(name+'.request')).touch(); stopped_at.append(clock[0])
        if clock[0] >= 1_500_000_000: queue.append(dict(type='STOP'))
    controller = CommittedTemporalController(.01, Mock())
    module.run(Node(), dict(temporal_ensemble_coeff=.01, n_action_steps=10), Mock(),
               now=lambda: clock[0], sleep=sleep, controller=controller)
    assert sent and all(t < stopped_at[0] for t, _ in sent)
    assert controller.block is None and controller.fusion.pending is None


def test_commit10_configuration_and_factory(tmp_path, monkeypatch):
    from deployment import chunk_execution
    from deployment.temporal_execution import make_controller
    (tmp_path/'assets').mkdir()
    config = dict(mode='act_temporal', n_action_steps=10, temporal_ensemble_coeff=.04743340266055522)
    (tmp_path/'assets/deployment.json').write_text(json.dumps(dict(action_execution=config)))
    monkeypatch.setattr(chunk_execution, 'ROOT', tmp_path)
    assert isinstance(make_controller(chunk_execution.execution_settings(), Mock()), CommittedTemporalController)
