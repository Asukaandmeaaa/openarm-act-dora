"""One prediction ahead; time-aligned finite segments, not ACT ensembling.

The current segment is never replaced. A prefetch credit authorizes exactly
one inference. The Dora actions queue stores that one result until completion.
No hardware imports or trajectory generation independent of ACT targets.
"""
import asyncio
import time
import numpy as np
import pyarrow as pa
from dora_openarm_actions_executor.main import BiquadLowpass, HermiteUpsampler, QPOS_TYPE
from .chunk_execution import prepare_chunk
from .common import session, request, write_json


class PipelineRequestGate:
    def __init__(self):
        self.pending = None
        self.credit = True
        self.fresh_after = -1

    def accept(self, event):
        if event.get('id') == 'chunk_prefetch':
            m = event['metadata']
            if self.pending is not None and m['chunk_id'] == self.pending and not self.credit:
                self.credit = True
                self.fresh_after = int(m['requested_monotonic_ns'])
            return False
        if event.get('id') != 'observation' or not self.credit:
            return False
        m = event['metadata']
        observed = int(m['observation_monotonic_ns'])
        oldest = observed - int(m.get('source_age_ns', 0))
        return observed > self.fresh_after and (self.fresh_after < 0 or oldest > self.fresh_after)

    def dispatched(self, chunk_id):
        if not self.credit:
            raise RuntimeError('only one prefetched chunk is allowed')
        self.pending = chunk_id
        self.credit = False


def aligned_chunk(event, begin_ns, previous, transition_ms):
    """Consume elapsed prediction prefix, blend within remaining time, keep tail.

Initial chunk retains all 50 slots. Subsequent chunks are anchored to the
observation assembly timestamp (source skew/age remains logged, not hidden).
The blend joins a held last emitted reference to the moving ACT trajectory;
it is not an appended transition or a new task waypoint.
"""
    _, _, full_duration = prepare_chunk(event, 50)  # shape/finite/interval contract
    m = event['metadata']
    observed = int(m['observation_monotonic_ns'])
    if observed > begin_ns:
        raise ValueError('observation timestamp is in the future')
    initial = previous is None
    age = 0 if initial else begin_ns - observed
    duration = full_duration - age
    # Do not rush a nearly expired tail (or truncate the transition to fit it).
    required = max(m['interval'], transition_ms * 1_000_000)
    if duration < required:
        return None
    offsets = np.arange(0, duration, 4_000_000, dtype=np.int64)
    q = np.asarray(event['value'].to_pylist(), dtype=np.float32)
    horizon = 49*m['interval']/1e9
    times = np.minimum((age + offsets)/1e9, horizon)
    values = HermiteUpsampler(1e9/m['interval'], horizon).upsample(q, times)
    transition_ns = 0 if initial else transition_ms * 1_000_000
    if transition_ns:
        u = np.clip(offsets/transition_ns, 0, 1)
        weight = (10*u**3 - 15*u**4 + 6*u**5)[:, None]
        values = np.asarray(previous)[None, :] + weight*(values - np.asarray(previous)[None, :])
    return offsets, values, duration, age, transition_ns


async def execute_pipeline_chunk(node, event, settings, lowpass, previous, stopped, log,
                                 *, now=time.monotonic_ns, sleep=asyncio.sleep):
    begin = now()
    if stopped():
        return previous, False
    m = event['metadata']
    plan = aligned_chunk(event, begin, previous, settings['transition_ms'])
    def prefetch():
        if stopped():
            return
        stamp = now()
        node.send_output('chunk_prefetch', pa.array([True]),
            {'chunk_id': m['chunk_id'], 'requested_monotonic_ns': stamp})
        log('chunk_prefetch_requested', chunk_id=m['chunk_id'], requested_monotonic_ns=stamp)
    if plan is None:
        log('chunk_expired', chunk_id=m['chunk_id'], observation_monotonic_ns=m['observation_monotonic_ns'])
        prefetch()  # hold; request a fresh prediction instead of replaying stale targets
        return previous, True
    offsets, values, duration, age, transition_ns = plan
    if previous is None or m.get('reset', False):
        lowpass.reset_state(values[0])
    elif transition_ns:
        # A held target has zero velocity; do not revive stale filter momentum.
        lowpass.reset_state(previous)
    log('chunk_started', chunk_id=m['chunk_id'], n_action_steps=50,
        execution_started_monotonic_ns=begin, duration_ns=duration,
        observation_age_ns=begin-int(m['observation_monotonic_ns']),
        skipped_prefix_ns=age, first_coarse_step=age//m['interval'],
        transition_ns=transition_ns, samples=len(offsets), execution_mode='pipelined')
    prefetched = False
    last = previous
    for index, (offset, value) in enumerate(zip(offsets, values)):
        delay = (begin + int(offset) - now())/1e9
        if delay > 0:
            await sleep(delay)
        if stopped():
            log('chunk_interrupted', chunk_id=m['chunk_id'], sample=index)
            return last, False
        if not prefetched and int(offset) >= duration-settings['prefetch_ms']*1_000_000:
            prefetch()
            prefetched = True
            if stopped():
                log('chunk_interrupted', chunk_id=m['chunk_id'], sample=index)
                return last, False
        position = lowpass.step(value)
        if not np.isfinite(position).all():
            raise ValueError('nonfinite pipelined reference')
        for side, sl in [('right', slice(0, 8)), ('left', slice(8, 16))]:
            node.send_output('move_position_'+side,
                pa.array([{'qpos': position[sl]}], type=QPOS_TYPE),
                {**m, 'timestamp': time.time_ns(), 'execution_mode': 'pipelined',
                 'n_action_steps': 50, 'coarse_step': min((age+int(offset))//m['interval'], 49),
                 'skipped_prefix_ns': age, 'transition_ns': transition_ns,
                 'reference_dt_ns': 4_000_000, 'reference_reset': index == 0,
                 'reference_end': index == len(values)-1})
        last = np.array(position, copy=True)
    delay = (begin+duration-now())/1e9
    if delay > 0:
        await sleep(delay)
    if stopped():
        return last, False
    if not prefetched:
        prefetch()
    completed = now()
    node.send_output('chunk_done', pa.array([True]),
        {'chunk_id': m['chunk_id'], 'completed_monotonic_ns': completed})
    log('chunk_completed', chunk_id=m['chunk_id'], completed_monotonic_ns=completed,
        elapsed_ns=completed-begin)
    return last, True


async def run(node, settings, log):
    path = session()
    def stopped():
        return any((path/(name+'.request')).exists() for name in ('stop', 'disable'))
    lowpass = BiquadLowpass(fs=250, fc=15)
    previous = None
    last_id = -1
    while True:
        if node.is_empty():
            await asyncio.sleep(.001)
            continue
        event = node.next()
        if event is None or event['type'] == 'STOP':
            return
        if event['type'] != 'INPUT' or event['id'] != 'actions' or stopped():
            continue
        try:
            cid = int(event['metadata']['chunk_id'])
            if cid <= last_id:
                log('chunk_duplicate_ignored', chunk_id=str(cid))
                continue
            last_id = cid
            previous, _ = await execute_pipeline_chunk(
                node, event, settings, lowpass, previous, stopped, log)
        except Exception as exc:
            request('stop', 'pipeline executor: '+str(exc))
            write_json(path/'executor_fault.json', {'reason': str(exc)})
            raise


def main(node, settings, log):
    asyncio.run(run(node, settings, log))
