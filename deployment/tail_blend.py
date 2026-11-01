"""Operator-selected two-prediction blend for the existing 100/50 scheduler.

Align by executed slots: the previous raw prediction's rows 50..99 pair with
the current raw prediction's rows 0..49. Observation/inference holds do not
advance this index. This is not wall-clock alignment or a temporal ensemble.
"""
import math

import numpy as np


def validate_new_weight(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('tail_blend_new_weight must be a finite number in [0, 1]')
    return float(value)


class PreviousTailBlender:
    def __init__(self, new_weight=None):
        self.new_weight = None if new_weight is None else validate_new_weight(new_weight)
        self.clear()

    def clear(self):
        self.tail = None
        self.chunk_id = None

    def apply(self, positions, *, chunk_id, reset=False):
        raw = np.asarray(positions, dtype=np.float32)
        if raw.shape != (100, 16) or not np.isfinite(raw).all():
            raise ValueError('tail blend requires finite 100x16 predictions')
        if reset:
            self.clear()
        result = raw.copy()
        used = self.new_weight is not None and self.tail is not None
        previous_id = self.chunk_id if used else None
        if used:
            result[:50] = (1.0-self.new_weight)*self.tail + self.new_weight*raw[:50]
        info = dict(enabled=self.new_weight is not None, applied=used,
                    configured_new_weight=self.new_weight,
                    effective_new_weight=self.new_weight if used else 1.0,
                    effective_old_weight=1.0-self.new_weight if used else 0.0,
                    previous_chunk_id=previous_id,
                    alignment='executed_slots_previous_50_99_to_current_0_49',
                    history='previous_raw_prediction_only')
        # Retain raw, unexecuted future targets. Never recursively blend history
        # or repeat the old executed prefix. Do not modify the input array.
        if self.new_weight is not None:
            self.tail = raw[50:].copy()
            self.chunk_id = chunk_id
        return result, info
