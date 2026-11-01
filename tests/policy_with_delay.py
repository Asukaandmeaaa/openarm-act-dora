"""Smoke-only real policy server: inject delay after the second inference.

Launched only by scripts/smoke_dataflow.py with fake sensors/motors. Production
entry points do not read this test delay setting.
"""
import os
import time
from deployment import policy

original_log = policy.Log
def delayed_log(role):
    log = original_log(role)
    count = 0
    def emit(event, **fields):
        nonlocal count
        log(event, **fields)
        if event == 'prediction':
            count += 1
            if count == 2:
                delay_ms = int(os.environ['TEST_POLICY_DELAY_MS'])
                log('test_inference_delay_begin', delay_ms=delay_ms)
                time.sleep(delay_ms/1000)
                log('test_inference_delay_end', delay_ms=delay_ms)
    return emit

if __name__ == '__main__':
    policy.Log = delayed_log
    policy.main()
