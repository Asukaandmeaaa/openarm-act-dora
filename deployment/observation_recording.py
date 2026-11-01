"""Bounded background copies of the exact Arrow files sent to the policy.

No sensor access or preprocessing. Recording failures never gate inference.
"""
import atexit
import json
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

from .common import session


class ObservationSnapshots:
    def __init__(self, path=None, *, interval_s=1.0, max_samples=120):
        self.path = (session() if path is None else Path(path)) / 'observations'
        self.interval_ns = int(interval_s * 1e9)
        self.max_samples = max_samples
        self.last_ns = None
        self.accepted = 0
        self.pending = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.worker = None
        try:
            self.path.mkdir(exist_ok=True)
        except OSError as exc:
            print(f'Observation recording unavailable: {exc}', file=sys.stderr)
            return
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        atexit.register(self.close)

    def _record(self, event, **data):
        try:
            with (self.path / 'index.jsonl').open('a') as f:
                f.write(json.dumps(dict(event=event, **data), default=str) + '\n')
        except OSError as exc:
            print(f'Observation index write failed: {exc}', file=sys.stderr)

    def capture(self, source, observation_id, metadata):
        now = time.monotonic_ns()
        if (self.worker is None or self.closed.is_set()
                or (self.max_samples is not None and self.accepted >= self.max_samples)
                or (self.last_ns is not None and now - self.last_ns < self.interval_ns)):
            return
        self.last_ns = now
        # Open before queuing: the official client unlinks older temporary
        # files after five requests. The descriptor keeps this exact file alive.
        try:
            stream = open(source, 'rb')
        except OSError as exc:
            self._record('capture_failed', observation_id=observation_id, error=str(exc))
            return
        item = (stream, self.accepted, observation_id, now, dict(metadata))
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            stream.close()
            self._record('capture_skipped_busy', observation_id=observation_id, metadata=metadata)
            return
        self.accepted += 1

    def _run(self):
        while not self.closed.is_set() or not self.pending.empty():
            try:
                stream, number, observation_id, now, metadata = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            target = self.path / f'{number:04d}.arrow'
            temporary = target.with_suffix('.arrow.partial')
            try:
                with stream, temporary.open('xb') as output:
                    shutil.copyfileobj(stream, output)
                temporary.replace(target)
                self._record('observation_saved', file=target.name,
                             observation_id=observation_id, captured_monotonic_ns=now,
                             metadata=metadata)
            except Exception as exc:
                # Source descriptors must close even when destination creation fails.
                stream.close()
                self._record('save_failed', observation_id=observation_id, error=str(exc))
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            finally:
                self.pending.task_done()

    def close(self):
        self.closed.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
