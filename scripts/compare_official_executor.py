"""Saved-chunk numerical comparison against pristine official source; no devices.

Uses a common virtual arrival schedule for both implementations. This proves
numerical parity under that schedule, not equivalence of real Dora queue timing.
"""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deployment.common import digest


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(module,rows,stop_ns):
    t0=rows[0]['monotonic_ns'];clock=SimpleNamespace(ns=0)
    class Done(Exception):pass
    class Events:
        index=0
        async def get(self):
            if self.index==len(rows):raise Done()
            row=rows[self.index];self.index+=1
            clock.ns=max(clock.ns,row['monotonic_ns']-t0)
            return {'value':pa.array(row['positions'],type=pa.list_(pa.float32())),
                    'metadata':row['metadata']}
        def empty(self):
            return self.index==len(rows) or rows[self.index]['monotonic_ns']-t0>clock.ns
    times=[];values=[]
    class Node:
        def send_output(self,name,value,metadata):
            if name=='move_position_right' and clock.ns<=stop_ns-t0:
                times.append(clock.ns);values.append(value.field('qpos')[0].as_py())
    async def sleep(seconds):clock.ns+=round(seconds*1e9)
    oldtime,oldasync=module.time,module.asyncio
    module.time=SimpleNamespace(time_ns=lambda:clock.ns)
    module.asyncio=SimpleNamespace(sleep=sleep)
    try:
        asyncio.run(module._main_executor(Node(),Events(),['right','left'],True,True,250))
    except Done:pass
    finally:module.time,module.asyncio=oldtime,oldasync
    return np.array(times),np.array(values)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    begin=json.loads((args.session/'start.request').read_text())['monotonic_ns']
    end=json.loads((args.session/'stop.request').read_text())['monotonic_ns']
    rows=[r for line in (args.session/'policy_client.jsonl').open()
          if begin<=(r:=json.loads(line))['monotonic_ns']<=end]
    relative=Path('dora-openarm-evaluation/nodes/dora-openarm-actions-executor/src/dora_openarm_actions_executor/main.py')
    deployed=ROOT/'upstream'/relative
    repo=next(r for r in json.loads((ROOT/'sources.lock.json').read_text())['repositories']
              if r['path'].endswith('/dora-openarm-actions-executor'))
    pristine=subprocess.check_output(['git','-C',str(ROOT/repo['path']),'show',
        repo['commit']+':src/dora_openarm_actions_executor/main.py'])
    with tempfile.TemporaryDirectory() as temporary:
        original=Path(temporary)/'official.py'; original.write_bytes(pristine)
        official_hash=digest(original)
        ta,a=replay(load(original,'pristine_executor'),rows,end)
    tb,b=replay(load(deployed,'deployed_executor'),rows,end)
    assert np.array_equal(ta,tb)
    np.testing.assert_array_equal(a,b)
    report=dict(hardware_accessed=False,prediction_chunks=len(rows),right_outputs=len(a),
                equal_timestamps=True,equal_positions=True,max_position_difference_rad=float(np.max(np.abs(a-b))),
                official_sha256=official_hash,deployed_sha256=digest(deployed),
                session=str(args.session),
                limitation='Identical virtual arrivals, no Dora transport queues, no physical tracking or closed-loop success claim.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
