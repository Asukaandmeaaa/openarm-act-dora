"""Instrument official full-chunk inference; preprocessing/ACT unchanged."""
import importlib.util
import os
import sys
import time
from .common import UPSTREAM, MODEL, Log, session, write_json


def main():
    from .chunk_execution import execution_settings
    if execution_settings()['mode']=='act_temporal':
        # Small CPU image/state operations otherwise spawn a large OpenMP
        # pool beside the GPU and Dora nodes, spoiling the 33ms deadline.
        # Tensor/model/processor math and saved weights are unchanged.
        import torch
        torch.set_num_threads(1)
    os.environ['OPENARM_PRETRAINED_PATH']=str(MODEL)
    path=UPSTREAM/'src/local_policy_server.py'
    spec=importlib.util.spec_from_file_location('official_policy_server',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    log=Log('policy')
    original=module.infer
    def infer(policy,pre,post,observation):
        began=time.monotonic_ns()
        result=original(policy,pre,post,observation)
        log('prediction',observation_id=observation['id'].as_py(),
            state=observation['position'].as_py(), positions=result['positions'],
            inference_ms=(time.monotonic_ns()-began)/1e6)
        return result
    module.infer=infer
    # Official server removes existing sockets: refuse BEFORE entering it.
    if os.path.exists(os.environ['SOCKET']): raise ValueError('policy socket exists; refuse duplicate')
    write_json(session()/'policy_process.json',dict(pid=os.getpid(),monotonic_ns=time.monotonic_ns()))
    sys.argv=[str(path),os.environ['SOCKET']]
    module.main()
