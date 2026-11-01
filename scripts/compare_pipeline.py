"""Saved predictions, virtual clock only. No inference/devices/physical graph.

Counterfactual next observations are unavailable: the age sweep is explicit,
not a claim that these chunks were predicted from prefetched physical images.
"""
import asyncio
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.chunk_executor import execute_chunk
from deployment.pipeline_execution import execute_pipeline_chunk
from dora_openarm_actions_executor.main import BiquadLowpass


def replay(rows, age_ms, transition_ms, baseline=False):
    clock=[1_000_000_000]; previous=None; filt=BiquadLowpass(250,15)
    records=[]; bounds=[]
    class Node:
        def send_output(self,name,value,metadata):
            if name=='move_position_right':
                records.append((clock[0],value.field('qpos')[0].as_py(),metadata))
    async def sleep(delay):clock[0]+=round(delay*1e9)
    async def go():
        nonlocal previous
        for i,r in enumerate(rows):
            m={**r['metadata'],'reset':i==0,'chunk_id':str(i+1),
               'observation_monotonic_ns':clock[0]-int(age_ms*1e6)}
            event=dict(value=pa.array(r['positions'],type=pa.list_(pa.float32())),metadata=m)
            bounds.append(len(records))
            if baseline:
                await execute_chunk(Node(),event,50,filt,lambda:False,lambda *a,**k:None,
                                    now=lambda:clock[0],sleep=sleep)
                clock[0]+=83_635_092
            else:
                previous,_=await execute_pipeline_chunk(Node(),event,
                    dict(prefetch_ms=150,transition_ms=transition_ms),filt,previous,
                    lambda:False,lambda *a,**k:None,now=lambda:clock[0],sleep=sleep)
    asyncio.run(go())
    q=np.array([r[1] for r in records]);dv=np.diff(q,axis=0)/.004
    velocity=dv.copy()
    # Match actual reference_reset/end semantics, but separately report boundary jumps.
    for i in range(1,len(records)):
        if records[i][2]['reference_reset'] or records[i][2]['reference_end']:velocity[i-1]=0
    boundary=np.array([abs(q[i]-q[i-1]) for i in bounds[1:]])
    return dict(samples=len(q),max_abs_reference_velocity=np.max(abs(velocity[:,:7]),axis=0).tolist(),
        max_adjacent_position_change=np.max(abs(np.diff(q,axis=0)),axis=0).tolist(),
        max_boundary_position_jump=np.max(boundary,axis=0).tolist() if len(boundary) else [],
        gripper_min=float(q[:,7].min()),last_gripper=float(q[-1,7]),
        all_segments_reach_step49=all(records[(bounds+[len(records)])[i+1]-1][2]['coarse_step']==49 for i in range(len(bounds))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path,help='Saved policy_client.jsonl from a 50-step legacy experiment')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    path=args.log
    rows=[json.loads(s) for s in path.read_text().splitlines() if json.loads(s)['event']=='raw_policy_chunk']
    results={'baseline':replay(rows,0,0,True)}
    for age in (50,100,150,250):
        for transition in (0,100,150):
            results[f'age{age}_transition{transition}']=replay(rows,age,transition)
    release=json.loads((ROOT/'tests/fixtures/release50_trial_phio1fa8.json').read_text())['prediction']
    release_rows=[{'positions':release['positions'],'metadata':{'interval':33_333_333}}]*2
    releases={str(t):replay(release_rows,150,t) for t in (0,100,150)}
    report=dict(hardware_accessed=False,physical_validation_passed=False,source=str(path),chunks=len(rows),
        limitation='Same saved prediction chunks, assumed observation ages; no counterfactual image/state feedback.',
        comparisons=results,saved_release_comparisons=releases)
    out=args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
