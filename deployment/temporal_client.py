"""30Hz, latest-observation queries for the opt-in ACT temporal experiment.

One slot of fixed transport latency: prediction step j is used at query slot
j+1. All predictions have that same shift. Missed slots are logged, never
relabelled or run in a catch-up burst. No hardware access in this module.
"""
import json
import tempfile
import time
import pyarrow as pa
from .common import session

INTERVAL = 33_333_333


def run(node, io, shared_dir, log, *, now=time.monotonic_ns,
        sleep=time.sleep, stopped=None, snapshots=None):
    # PyArrow may lazily import pandas on its first Python-list conversion.
    # Pay that one-time cost before establishing the 30Hz epoch/deadlines.
    pa.array([[0.]*16], type=pa.list_(pa.float32()))
    owns_snapshots = False
    if stopped is None:
        path = session()
        stopped = lambda:any((path/(n+'.request')).exists() for n in ('stop', 'disable'))
        if snapshots is None:
            from .observation_recording import ObservationSnapshots
            snapshots = ObservationSnapshots(path=path, interval_s=0, max_samples=None)
            owns_snapshots = True
    files = []
    latest = None
    epoch = None
    next_step = 0
    previous_id = None
    try:
        while True:
            # Queue size one in the graph. Bound draining so an input flood
            # cannot starve either STOP checks or the scheduling clock.
            for _ in range(32):
                if node.is_empty():
                    break
                event = node.next()
                if event is None or event['type'] == 'STOP':
                    return
                if event['type'] == 'INPUT' and event['id'] == 'observation':
                    latest = event
            if stopped():
                latest = None
                sleep(.001)
                continue
            if latest is None:
                sleep(.001)
                continue
            if epoch is None:
                epoch = now()
            stamp = now()
            due = epoch + next_step*INTERVAL
            if stamp < due:
                sleep(min(.001, (due-stamp)/1e9))
                continue
            step = max(next_step, (stamp-epoch)//INTERVAL)
            if step > next_step:
                log('temporal_query_slots_skipped', first=next_step, count=step-next_step)
            next_step = step+1
            target = epoch+(step+1)*INTERVAL
            observation = latest['value']
            oid = observation.field('id')[0].as_py()
            reset = previous_id is None or oid < previous_id
            previous_id = oid
            metadata = {**latest['metadata'], 'temporal_epoch_ns': epoch,
                        'temporal_step': step, 'temporal_target_ns': target,
                        'query_started_monotonic_ns': stamp,
                        'chunk_id': f'{epoch}:{step}'}
            with tempfile.NamedTemporaryFile(suffix='.arrow', dir=shared_dir,
                                             delete=False) as f:
                files.append(f.name)
                batch = pa.RecordBatch.from_struct_array(observation)
                with pa.ipc.new_file(f, batch.schema) as writer:
                    writer.write(batch)
            if len(files) > 5:
                __import__('os').unlink(files.pop(0))
            request = dict(name='inference', data_path=files[-1], reset=reset, metadata=metadata)
            if snapshots is not None:
                snapshots.capture(files[-1], oid, {**metadata, 'reset': reset})
            io.write(json.dumps(request)+'\n')
            io.flush()
            response = io.readline()
            if not response:
                return
            actions = json.loads(response)
            ended = now()
            if stopped():
                log('prediction_discarded_after_stop', temporal_step=step)
                continue
            latency = (ended-stamp)/1e6
            if ended >= target:
                log('temporal_prediction_late', temporal_step=step, latency_ms=latency,
                    deadline_ns=target, completed_ns=ended)
                continue
            metadata.update(interval=actions['interval'], reset=reset,
                            cutoff_hz=actions.get('cutoff_hz',15))
            log('raw_policy_chunk', positions=actions['positions'], metadata=metadata,
                round_trip_ms=latency, deadline_margin_ms=(target-ended)/1e6)
            node.send_output('actions', pa.array(actions['positions'],type=pa.list_(pa.float32())), metadata)
    finally:
        if owns_snapshots:
            snapshots.close()
        import os
        for f in files:
            os.unlink(f)
