"""Explicit official-blend or complete-50-step experiment configuration."""
import json
import math
from pathlib import Path
import numpy as np
from .common import ROOT


def execution_settings():
    import os
    config_path = Path(os.environ.get('OPENARM_DEPLOYMENT_CONFIG', 'assets/deployment.json'))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text()).get(
        'action_execution', {'mode': 'official_blend'})
    mode = config.get('mode')
    if mode not in ('official_blend', 'chunked', 'pipelined', 'chunked_prefetch', 'act_temporal', 'chunk100_exec50'):
        raise ValueError('unknown action execution mode')
    if mode == 'act_temporal':
        chunk_size = config.get('chunk_size', 50)
        if type(chunk_size) is not int or chunk_size not in (50, 100):
            raise ValueError('ACT temporal chunk_size must be 50 or 100')
        coefficient = config.get('temporal_ensemble_coeff')
        if type(coefficient) not in (int, float) or not math.isfinite(coefficient):
            raise ValueError('ACT temporal coefficient must be finite')
        # LeRobot supports signed coefficients: negative favors newer predictions.
        # Its online accumulator uses float32, so reject overflowing weights.
        with np.errstate(over='ignore', invalid='ignore'):
            weights = np.exp(-coefficient * np.arange(chunk_size, dtype=np.float64))
            weight_sum = weights.astype(np.float32).sum(dtype=np.float32)
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError('ACT temporal coefficient overflows float32 weights')
        allowed_steps = (1, 10) if chunk_size == 50 else (1, 50)
        if type(config.get('n_action_steps')) is not int or config['n_action_steps'] not in allowed_steps:
            raise ValueError(f'ACT temporal mode requires n_action_steps=1 or {allowed_steps[-1]} for chunk_size={chunk_size}')
        if any(k in config for k in ('prefetch_ms', 'transition_ms', 'tail_blend_new_weight')):
            raise ValueError('ACT temporal mode does not use segment prefetch/transition')
        return config
    if mode == 'chunk100_exec50':
        if type(config.get('n_action_steps')) is not int or config['n_action_steps'] != 50:
            raise ValueError('chunk100_exec50 requires n_action_steps=50')
        if type(config.get('chunk_size')) is not int or config['chunk_size'] != 100:
            raise ValueError('chunk100_exec50 requires chunk_size=100')
        if any(k in config for k in ('prefetch_ms', 'transition_ms', 'temporal_ensemble_coeff')):
            raise ValueError('chunk100_exec50 does not use prefetch/transition/ensemble')
        if 'tail_blend_new_weight' in config:
            from .tail_blend import validate_new_weight
            validate_new_weight(config['tail_blend_new_weight'])
        return config
    if config.get('temporal_ensemble_coeff') is not None:
        raise ValueError('temporal ensembling is not enabled in this experiment')
    if mode in ('chunked', 'pipelined', 'chunked_prefetch'):
        if type(config.get('n_action_steps')) is not int or config['n_action_steps'] != 50:
            raise ValueError('this experiment requires n_action_steps=50')
    elif 'n_action_steps' in config:
        raise ValueError('official_blend does not use a fixed execution step count; remove n_action_steps')
    if mode == 'pipelined':
        for key, bounds in [('prefetch_ms', (1, 500)), ('transition_ms', (0, 300))]:
            value = config.get(key)
            if type(value) is not int or not bounds[0] <= value <= bounds[1]:
                raise ValueError('invalid pipelined '+key)
    if mode == 'chunked_prefetch':
        value = config.get('prefetch_ms')
        if type(value) is not int or not 1 <= value <= 500:
            raise ValueError('invalid chunked_prefetch prefetch_ms')
        if 'transition_ms' in config:
            raise ValueError('chunked_prefetch has no transition; remove transition_ms')
    return config


class ChunkRequestGate:
    def __init__(self, enabled):
        self.enabled = enabled
        self.pending = None
        self.completed_ns = -1

    def accept(self, event):
        if event.get('id') == 'chunk_done':
            metadata = event['metadata']
            if self.enabled and self.pending is not None and metadata.get('chunk_id') == self.pending:
                self.completed_ns = int(metadata['completed_monotonic_ns'])
                self.pending = None
            return False
        if event.get('id') != 'observation':
            return False
        if not self.enabled:
            return True
        metadata = event['metadata']
        observed = int(metadata['observation_monotonic_ns'])
        # After completion, cached camera/arm samples must also have advanced.
        # No fixed age threshold: this compares against this chunk's completion.
        oldest_source = observed - int(metadata.get('source_age_ns', 0))
        return (self.pending is None and observed > self.completed_ns and
                (self.completed_ns < 0 or oldest_source > self.completed_ns))

    def dispatched(self, chunk_id):
        if self.enabled:
            if self.pending is not None:
                raise RuntimeError('cannot replace a committed action chunk')
            self.pending = chunk_id


def prepare_chunk(event, steps, chunk_size=50):
    from dora_openarm_actions_executor.main import HermiteUpsampler
    q = np.asarray(event['value'].to_pylist(), dtype=np.float32)
    if q.shape != (chunk_size, 16) or not np.isfinite(q).all():
        raise ValueError(f'expected finite {chunk_size}x16 ACT chunk')
    interval = event['metadata']['interval']
    if type(interval) is not int or interval != 33_333_333:
        raise ValueError('chunk experiment requires training interval 33333333ns')
    if type(steps) is not int or not 2 <= steps <= chunk_size:
        raise ValueError('invalid committed action count')
    dt = interval/1e9
    # N coarse targets occupy N 30Hz slots. Hold the final target in the
    # last slot; do not interpolate toward any discarded tail action.
    sample_ns = np.arange(0, steps*interval, 4_000_000, dtype=np.int64)
    times = np.minimum(sample_ns/1e9, (steps-1)*dt)
    upsampler = HermiteUpsampler(1/dt, (steps-1)*dt)
    values = upsampler.upsample(q[:steps], times)
    return sample_ns, values, steps*interval
