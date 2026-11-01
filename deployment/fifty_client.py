"""Local policy client for predict-100 / execute-first-50.

The official client's ChunkRequestGate is enabled only for its own 'chunked'
mode.  This local client implements the same request/response protocol with
the gate always enabled, so a new observation is requested only after the
executor reports completion of the committed 50 steps. Optional two-prediction
tail blending occurs before official interpolation/filtering; raw predictions
and the actual committed prefix are both logged. Upstream sources are unchanged.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time

import dora
import pyarrow as pa

from .chunk_execution import ChunkRequestGate, execution_settings
from .common import Log, session
from .observation_recording import ObservationSnapshots
from .tail_blend import PreviousTailBlender


def main() -> None:
    execution = execution_settings()
    if execution["mode"] != "chunk100_exec50":
        raise ValueError("fifty_client is only for chunk100_exec50")
    chunk_size = int(execution["chunk_size"])
    n_action_steps = int(execution["n_action_steps"])
    node = dora.Node()
    log = Log("policy_client")
    gate = ChunkRequestGate(True)
    blender = PreviousTailBlender(execution.get("tail_blend_new_weight"))
    snapshots = ObservationSnapshots(path=session(), interval_s=0, max_samples=None)
    previous_observation_id = None
    data_files = []
    shared_dir = tempfile.mkdtemp(prefix="act100-50-", dir="/dev/shm")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(os.environ["SOCKET"])
    io = sock.makefile("rw")

    def stopped() -> bool:
        return any((session() / f"{name}.request").exists() for name in ("stop", "disable"))

    try:
        for event in node:
            if event["type"] != "INPUT":
                continue
            if stopped():
                gate.pending = None
                blender.clear()
                log("observation_ignored_after_stop")
                continue
            if not gate.accept(event):
                continue
            observation = event["value"]
            observation_id = observation.field("id")[0].as_py()
            reset = previous_observation_id is None or observation_id < previous_observation_id
            previous_observation_id = observation_id
            data_file = tempfile.NamedTemporaryFile(
                suffix=".arrow", dir=shared_dir, delete_on_close=False
            )
            record_batch = pa.RecordBatch.from_struct_array(observation)
            with pa.output_stream(data_file) as output:
                with pa.ipc.new_file(output, record_batch.schema) as writer:
                    writer.write(record_batch)
            data_files.append(data_file)
            if len(data_files) > 5:
                data_files.pop(0)
            request = {
                "name": "inference",
                "data_path": data_file.name,
                "reset": reset,
                "metadata": event["metadata"],
            }
            snapshots.capture(
                request["data_path"],
                previous_observation_id,
                {**request["metadata"], "reset": reset},
            )
            requested = time.monotonic_ns()
            io.write(json.dumps(request) + "\n")
            io.flush()
            response = io.readline()
            if not response:
                raise RuntimeError("policy server closed the connection")
            reply = time.monotonic_ns()
            actions = json.loads(response)
            if stopped():
                gate.pending = None
                blender.clear()
                log("prediction_discarded_after_stop", requested_ns=requested, replied_ns=reply)
                continue
            positions = actions.get("positions", [])
            if len(positions) != chunk_size or any(len(row) != 16 for row in positions):
                raise ValueError(
                    f"expected {chunk_size}x16 prediction, got {len(positions)} rows"
                )
            metadata = {
                "interval": actions["interval"],
                "reset": reset,
                "chunk_id": str(event["metadata"]["observation_monotonic_ns"]),
                "chunk_size": chunk_size,
                "n_action_steps": n_action_steps,
                "execution_mode": "chunk100_exec50",
                "requested_monotonic_ns": requested,
                "replied_monotonic_ns": reply,
                "inference_latency_ns": reply - requested,
            }
            metadata.update(event["metadata"])
            if "cutoff_hz" in actions:
                metadata["cutoff_hz"] = actions["cutoff_hz"]
            dispatched, blend_info = blender.apply(
                positions, chunk_id=metadata["chunk_id"], reset=reset
            )
            metadata["tail_blend_applied"] = blend_info["applied"]
            metadata["tail_blend_new_weight"] = blend_info["effective_new_weight"]
            metadata["tail_blend_old_weight"] = blend_info["effective_old_weight"]
            metadata["tail_blend_alignment"] = blend_info["alignment"]
            gate.dispatched(metadata["chunk_id"])
            log(
                "raw_policy_chunk",
                positions=positions,
                metadata=metadata,
                committed=dispatched[:n_action_steps].tolist(),
                trajectory_blend=blend_info,
            )
            node.send_output(
                "actions",
                pa.array(dispatched.tolist(), type=pa.list_(pa.float32())),
                metadata,
            )
    finally:
        blender.clear()
        snapshots.close()
        try:
            io.close()
            sock.close()
        finally:
            for f in data_files:
                try:
                    os.unlink(f.name)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
