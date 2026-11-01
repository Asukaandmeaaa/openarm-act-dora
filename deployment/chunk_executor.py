"""Operator-selected finite chunk scheduler using official interpolation/filter.

Optional old/new tail blending is performed by the client before this module.
No temporal ensemble here. Drivers and operator STOP remain the sole hardware
path; this module only emits Dora references.
"""
import asyncio
import time
import numpy as np
import pyarrow as pa
from dora_openarm_actions_executor.main import BiquadLowpass, QPOS_TYPE
from .common import session, request, write_json
from .chunk_execution import prepare_chunk


async def execute_chunk(node, event, steps, lowpass, stopped, log,
                        *, now=time.monotonic_ns, sleep=asyncio.sleep, prefetch_ms=0, chunk_size=50, mode_name=None):
    if stopped():
        return False
    sample_ns, values, duration_ns = prepare_chunk(event, steps, chunk_size)
    metadata = event['metadata']
    if metadata.get('reset', False):
        lowpass.reset_state(values[0])
    begin = now()
    mode = mode_name or ('chunked_prefetch' if prefetch_ms else 'chunked')
    log('chunk_started', chunk_id=metadata['chunk_id'], n_action_steps=steps, chunk_size=chunk_size,
        duration_ns=duration_ns, temporal_ensemble_coeff=None,
        execution_mode=mode, skipped_prefix_ns=0, first_coarse_step=0,
        observation_age_ns=begin-int(metadata.get('observation_monotonic_ns', begin)),
        execution_started_monotonic_ns=begin)
    prefetched = False
    for index, (offset, value) in enumerate(zip(sample_ns, values)):
        delay = (begin + int(offset) - now())/1e9
        if delay > 0:
            await sleep(delay)
        if stopped():
            log('chunk_interrupted', chunk_id=metadata['chunk_id'], sample=index)
            return False
        if prefetch_ms and not prefetched and int(offset) >= duration_ns-prefetch_ms*1_000_000:
            stamp = now()
            node.send_output('chunk_prefetch', pa.array([True]),
                {'chunk_id': metadata['chunk_id'], 'requested_monotonic_ns': stamp})
            log('chunk_prefetch_requested', chunk_id=metadata['chunk_id'], requested_monotonic_ns=stamp)
            prefetched = True
            if stopped():
                log('chunk_interrupted', chunk_id=metadata['chunk_id'], sample=index)
                return False
        position = lowpass.step(value)
        if not np.isfinite(position).all():
            raise ValueError('nonfinite filtered chunk reference')
        for side, sl in [('right', slice(0, 8)), ('left', slice(8, 16))]:
            node.send_output('move_position_'+side,
                pa.array([{'qpos': position[sl]}], type=QPOS_TYPE),
                {**metadata, 'timestamp': time.time_ns(),
                 'execution_mode': mode, 'n_action_steps': steps,
                 'skipped_prefix_ns': 0,
                 'coarse_step': min(int(offset)//metadata['interval'], steps-1),
                 'reference_dt_ns': 4_000_000,
                 # The next chunk follows an observation/inference gap. Do not
                 # differentiate across that gap as if it were just 4ms.
                 'reference_reset': index == 0,
                 'reference_end': index == len(values)-1})
    delay = (begin + duration_ns - now())/1e9
    if delay > 0:
        await sleep(delay)
    if stopped():
        return False
    completed = now()
    node.send_output('chunk_done', pa.array([True]),
        {'chunk_id': metadata['chunk_id'], 'completed_monotonic_ns': completed})
    log('chunk_completed', chunk_id=metadata['chunk_id'],
        completed_monotonic_ns=completed, elapsed_ns=completed-begin)
    return True


async def run(node, steps, log, *, prefetch_ms=0, chunk_size=50, mode_name=None):
    path = session()
    def stopped():
        return any((path/(name+'.request')).exists() for name in ('stop', 'disable'))
    lowpass = BiquadLowpass(fs=250, fc=15)
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
            if prefetch_ms:
                cid = int(event['metadata']['chunk_id'])
                if cid <= last_id:
                    log('chunk_duplicate_ignored', chunk_id=str(cid))
                    continue
                last_id = cid
            await execute_chunk(node, event, steps, lowpass, stopped, log, prefetch_ms=prefetch_ms, chunk_size=chunk_size, mode_name=mode_name)
        except Exception as exc:
            request('stop', 'chunk executor: '+str(exc))
            write_json(path/'executor_fault.json', {'reason': str(exc)})
            raise


def main(node, steps, log, *, prefetch_ms=0, chunk_size=50, mode_name=None):
    asyncio.run(run(node, steps, log, prefetch_ms=prefetch_ms, chunk_size=chunk_size, mode_name=mode_name))
