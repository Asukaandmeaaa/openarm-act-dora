"""Freeze an explicit block of aligned targets while 30Hz inference continues.

The original TemporalController maintains the unmodified LeRobot ensemble.
Only at a block boundary do we copy its next targets into an immutable
execution block. Incoming predictions continue updating the future candidate
but cannot replace any target in the active block. No hardware is opened here.
"""
import time
import numpy as np
from dora_openarm_actions_executor.main import BiquadLowpass
from .temporal_client import INTERVAL
from .temporal_execution import TemporalController


class CommittedTemporalController:
    steps = 10

    def __init__(self, coefficient, log, *, chunk_size=50, steps=10):
        if type(steps) is not int or (chunk_size, steps) not in ((50, 10), (100, 50)):
            raise ValueError('committed temporal requires 50/10 or 100/50')
        self.steps = steps
        self.chunk_size = chunk_size
        self.coefficient = coefficient
        self.log = log
        def fusion_log(event, **fields):
            if event == 'temporal_step_applied':
                event = 'temporal_ensemble_updated'
            log(event, **fields)
        self.fusion = TemporalController(coefficient, fusion_log, chunk_size=chunk_size)
        self.lowpass = BiquadLowpass(250, 15)
        self.clear()

    def clear(self):
        self.fusion.clear()
        self.block = None
        self.output = self.goal = self.metadata = None
        self.last_fused_key = None
        self.block_index = -1
        self.expires = 0
        self.holding = False
        self.reset_reference = False

    def accept(self, event, now):
        self.fusion.accept(event, now)

    def activate(self, now):
        pending = self.fusion.pending
        self.fusion.activate(now)
        fresh = self.fusion.last_key != self.last_fused_key
        if fresh:
            self.last_fused_key = self.fusion.last_key
        # A fresh, on-time prediction is required to start every block.
        # During the block, even a missed inference cannot rewrite its targets.
        if fresh and (self.block is None or now >= self.expires):
            target = self.fusion.metadata['temporal_target_ns']
            first = self.fusion.goal[None]
            remaining = self.fusion.ensemble.ensembled_actions.numpy()[0]
            self.block = np.concatenate((first, remaining[:self.steps-1])).copy()
            counts = self.fusion.ensemble.ensembled_actions_count.numpy()[:, 0]
            self.counts = np.r_[self.fusion.count, counts[:self.steps-1]].copy()
            # Official per-target normalization: future targets have fewer
            # contributing predictions. Log actual weights, not just the 1%
            # full-history block-start value.
            weights = self.fusion.ensemble.ensemble_weights.numpy()
            sums = self.fusion.ensemble.ensemble_weights_cumsum.numpy()
            self.latest_weights = (weights[self.counts.astype(int)-1] /
                                   sums[self.counts.astype(int)-1]).copy()
            self.raw_block = pending[2][:self.steps].copy()
            self.block_metadata = dict(self.fusion.metadata)
            self.block_start = target
            self.expires = target+self.steps*INTERVAL
            self.block_index = -1
            self.reset_reference = self.output is None or self.holding
            if self.reset_reference and self.output is not None:
                self.lowpass.reset_state(self.output)
            self.holding = False
            self.log('temporal_block_committed',
                     temporal_step=self.fusion.last_key[1], target_ns=target,
                     expires_ns=self.expires, n_action_steps=self.steps,
                     targets=self.block, contributors=self.counts,
                     latest_weights=self.latest_weights, chunk_size=self.chunk_size,
                     temporal_ensemble_coeff=self.coefficient)
        if self.block is None or now >= self.expires:
            return
        index = (now-self.block_start)//INTERVAL
        if index == self.block_index:
            return
        if index != self.block_index+1:
            # A scheduler stall must not compress/skip through an old block.
            self.log('temporal_block_aborted_late', previous_index=self.block_index,
                     due_index=index, target_ns=self.block_start)
            self.block = None
            self.expires = now
            self.fusion.clear()
            self.last_fused_key = None
            return
        self.block_index = index
        self.goal = self.block[index].copy()
        if self.output is None:
            self.lowpass.reset_state(self.goal)
        step = self.block_metadata['temporal_step']+index
        target = self.block_start+index*INTERVAL
        self.metadata = {**self.block_metadata, 'temporal_step': step,
                         'temporal_target_ns': target,
                         'committed_source_step': self.block_metadata['temporal_step'],
                         'committed_step_index': index}
        self.log('temporal_step_applied', temporal_step=step, target_ns=target,
                 applied_ns=now, contributors=int(self.counts[index]),
                 raw_first=self.raw_block[index], raw_source_index=index,
                 fused_target=self.goal,
                 committed_source_step=self.block_metadata['temporal_step'])

    def sample(self, now):
        self.activate(now)
        if self.goal is None:
            return None
        ending = now >= self.expires
        if ending:
            if self.holding or self.output is None:
                return None
            self.holding = True
            self.log('temporal_wait_hold', temporal_step=self.metadata['temporal_step'])
        else:
            self.output = self.lowpass.step(self.goal)
        if not np.isfinite(self.output).all():
            raise ValueError('nonfinite committed temporal output')
        metadata = {**self.metadata, 'timestamp': time.time_ns(),
                    'execution_mode': 'act_temporal', 'n_action_steps': self.steps,
                    'chunk_size': self.chunk_size,
                    'reference_dt_ns': 4_000_000, 'reference_end': ending,
                    'reference_reset': self.reset_reference}
        self.reset_reference = False
        return self.output.copy(), metadata
