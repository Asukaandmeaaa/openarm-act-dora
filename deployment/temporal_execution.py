"""Standard LeRobot ACT ensemble, with explicitly numbered 30Hz time slots.

The ensemble operates on denormalized predictions: the frozen action
postprocessor is affine, hence it commutes with weighted averaging. No
recursive OpenArm tail blend, chunk-prefix skipping or boundary bridge.

Each fused target is held for one 30Hz slot, through the existing 250Hz/15Hz
filter. A missing prediction holds the last emitted position with zero dq;
the next accepted slot resets the ensemble rather than fusing wrong times.
"""
import time
import numpy as np
import pyarrow as pa
from dora_openarm_actions_executor.main import BiquadLowpass, QPOS_TYPE
from .common import session, request, write_json
from .temporal_client import INTERVAL


class TemporalController:
    def __init__(self, coefficient, log, chunk_size=50):
        # Lazy import: other execution modes do not load Torch in the executor.
        import torch
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        torch.set_num_threads(1)
        self.torch = torch
        pa.array([{'qpos':[0.]*8}],type=QPOS_TYPE)
        if type(chunk_size) is not int or chunk_size not in (50, 100):
            raise ValueError('ACT temporal chunk_size must be 50 or 100')
        self.chunk_size = chunk_size
        self.ensemble = ACTTemporalEnsembler(coefficient, chunk_size)
        self.lowpass = BiquadLowpass(250, 15)
        self.log = log
        self.pending = None
        self.last_key = None
        self.last_received = None
        self.goal = None
        self.output = None
        self.metadata = None
        self.expires = 0
        self.count = 0
        self.reset_reference = False
        self.holding = False

    def clear(self):
        self.ensemble.reset()
        self.pending = None
        self.goal = None
        self.output = None
        self.metadata = None
        self.last_key = self.last_received = None
        self.count = 0
        self.expires = 0
        self.holding = False

    def accept(self, event, now):
        m = event['metadata']
        epoch, step, target = (m[k] for k in ('temporal_epoch_ns','temporal_step','temporal_target_ns'))
        if (any(type(v) is not int for v in (epoch,step,target)) or epoch < 0 or step < 0
                or target != epoch+(step+1)*INTERVAL or m['interval'] != INTERVAL):
            raise ValueError('invalid ACT temporal clock metadata')
        key = (epoch,step)
        if self.last_received is not None and key <= self.last_received:
            self.log('temporal_duplicate_ignored', temporal_step=step)
            return
        if now >= target:
            self.log('temporal_transport_late', temporal_step=step, lateness_ns=now-target)
            return
        if target-now > INTERVAL:
            raise ValueError('ACT temporal prediction more than one slot in future')
        q = np.asarray(event['value'].to_pylist(), dtype=np.float32)
        if q.shape != (self.chunk_size,16) or not np.isfinite(q).all():
            raise ValueError(f'expected finite {self.chunk_size}x16 ACT temporal prediction')
        if self.pending is not None:
            raise ValueError('ACT temporal single pending slot overflow')
        self.pending = (key, target, q, dict(m))
        self.last_received = key
        self.log('temporal_prediction_buffered', temporal_step=step, target_ns=target)

    def activate(self, now):
        if self.pending is None or now < self.pending[1]:
            return
        key, target, q, metadata = self.pending
        self.pending = None
        if now >= target+INTERVAL:
            self.log('temporal_execution_slot_missed', temporal_step=key[1])
            return
        consecutive = (self.last_key is not None and key == (self.last_key[0],self.last_key[1]+1))
        if not consecutive or metadata.get('reset',False) or self.holding:
            self.ensemble.reset()
            self.count = 0
            self.reset_reference = True
            if self.output is not None:
                self.lowpass.reset_state(self.output)
            self.log('temporal_ensemble_reset', temporal_step=key[1], initial=self.last_key is None)
        self.goal = self.ensemble.update(self.torch.from_numpy(q[None])).numpy()[0].copy()
        if self.output is None:
            self.lowpass.reset_state(self.goal)
        self.count = min(self.chunk_size, self.count+1)
        self.last_key = key
        self.metadata = metadata
        self.expires = target+INTERVAL
        self.holding = False
        self.log('temporal_step_applied', temporal_step=key[1], target_ns=target,
                 applied_ns=now, contributors=self.count, raw_first=q[0], fused_target=self.goal)

    def sample(self, now):
        self.activate(now)
        if self.goal is None:
            return None
        ending = now >= self.expires
        if ending:
            if self.holding or self.output is None:
                return None
            self.holding = True
            self.log('temporal_wait_hold', temporal_step=self.last_key[1])
        else:
            self.output = self.lowpass.step(self.goal)
        if not np.isfinite(self.output).all():
            raise ValueError('nonfinite ACT temporal output')
        metadata = {**self.metadata, 'timestamp': time.time_ns(),
                    'execution_mode': 'act_temporal', 'n_action_steps': 1,
                    'reference_dt_ns': 4_000_000, 'reference_end': ending,
                    'reference_reset': self.reset_reference}
        self.reset_reference = False
        return self.output.copy(), metadata


def make_controller(settings, log):
    if settings.get('n_action_steps', 1) in (10, 50):
        from .temporal_commit import CommittedTemporalController
        return CommittedTemporalController(settings['temporal_ensemble_coeff'], log,
            chunk_size=settings.get('chunk_size', 50), steps=settings['n_action_steps'])
    return TemporalController(settings['temporal_ensemble_coeff'], log,
                              chunk_size=settings.get('chunk_size', 50))


def run(node, settings, log, *, now=time.monotonic_ns, sleep=time.sleep, stopped=None, controller=None):
    path = session()
    if stopped is None:
        stopped = lambda:any((path/(n+'.request')).exists() for n in ('stop','disable'))
    if controller is None:
        controller = make_controller(settings, log)
    next_tick = now()
    try:
        while True:
            if stopped():
                controller.clear()
            controller.activate(now())
            for _ in range(32):
                if node.is_empty():
                    break
                event = node.next()
                if event is None or event['type']=='STOP':
                    controller.clear()
                    return
                if event['type']=='INPUT' and event['id']=='actions' and not stopped():
                    controller.activate(now())
                    controller.accept(event, now())
            stamp = now()
            if stamp >= next_tick:
                skipped = (stamp-next_tick)//4_000_000
                if skipped:
                    log('temporal_output_ticks_skipped', count=skipped)
                    controller.reset_reference = True
                next_tick += (skipped+1)*4_000_000
                if not stopped():
                    result = controller.sample(stamp)
                    if result is not None:
                        q, metadata = result
                        for side, sl in [('right',slice(0,8)), ('left',slice(8,16))]:
                            if stopped():
                                controller.clear()
                                break
                            node.send_output('move_position_'+side,
                                pa.array([{'qpos':q[sl]}], type=QPOS_TYPE), metadata)
            sleep(min(.001,max(0,(next_tick-now())/1e9)))
    except Exception as exc:
        controller.clear()
        request('stop','ACT temporal executor: '+str(exc))
        write_json(path/'executor_fault.json',{'reason':str(exc)})
        raise
