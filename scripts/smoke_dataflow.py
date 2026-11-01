"""Real policy + official components + TEST fake sensors/actuators, no hardware.

Never runs the production graph. A separate generated graph replaces ALL
robot/camera nodes with tests/fake_node.sh. Does not change results/current.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import yaml
import argparse
import threading

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.common import write_json
from deployment.chunk_execution import execution_settings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,help='write a new per-experiment report without replacing older evidence')
    parser.add_argument('--collection-feedforward',action='store_true')
    parser.add_argument('--exercise-stop',action='store_true',help='request stop/disable in the fake graph only')
    parser.add_argument('--saved-images',action='store_true',help='use saved initial demonstration images instead of black test images; no camera access')
    parser.add_argument('--inference-delay-ms',type=int,default=0,help='test-only second response delay; requires prefetch mode')
    args=parser.parse_args()
    execution=execution_settings()
    if not 0<=args.inference_delay_ms<=1000:parser.error('delay must be 0..1000ms')
    if args.inference_delay_ms and execution['mode'] not in ('pipelined','chunked_prefetch','act_temporal'):parser.error('delay test requires prefetch or temporal mode')
    if args.exercise_stop and not args.collection_feedforward: parser.error('--exercise-stop requires --collection-feedforward')
    (ROOT/'results').mkdir(exist_ok=True)
    path=Path(tempfile.mkdtemp(prefix='software_smoke_',dir=ROOT/'results'))
    env={**os.environ,'ACT_SESSION':str(path),'SOCKET':str(path/'policy.sock'),
         'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1'}
    if args.collection_feedforward: env['TEST_COLLECTION_FEEDFORWARD']='1'
    if args.saved_images:
        env['TEST_SAVED_IMAGES']=str(ROOT/'examples/images')
    graph=yaml.safe_load((ROOT/'dataflow.yml').read_text())
    for node in graph['nodes']:
        fake=node['id'] in ('robot-right','robot-left','camera-ceiling','camera-wrist')
        node['path']=str(ROOT/('tests/fake_node.sh' if fake else 'scripts/node'))
        if fake and node['args'].split()[0] not in ('robot','camera'):raise ValueError('unexpected fake mapping')
    assert sum(n['path']==str(ROOT/'tests/fake_node.sh') for n in graph['nodes'])==4
    graph_path=path/'fake_dataflow.yml';graph_path.write_text(yaml.safe_dump(graph,sort_keys=False))
    # These requests exist ONLY in the isolated graph with fake motor drivers.
    write_json(path/'enable.request',dict(monotonic_ns=time.monotonic_ns(),reason='software-only fake graph'))
    write_json(path/'start.request',dict(monotonic_ns=time.monotonic_ns(),reason='software-only fake graph'))
    with (path/'policy_stdout.txt').open('w') as out:
        command=[str(ROOT/'scripts/node'),'policy']
        if args.inference_delay_ms:
            env['TEST_POLICY_DELAY_MS']=str(args.inference_delay_ms)
            command=[str(ROOT/'.venv/bin/python'),str(ROOT/'tests/policy_with_delay.py')]
        policy=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+90
            while not Path(env['SOCKET']).exists():
                if policy.poll() is not None:raise RuntimeError((path/'policy_stdout.txt').read_text())
                if time.monotonic()>deadline:raise TimeoutError('policy startup')
                time.sleep(.1)
            stop_errors=[]
            def stop_fake_graph():
                try:
                    deadline=time.monotonic()+22
                    while time.monotonic()<deadline:
                        try: moves=int((path/'fake_moves.txt').read_text())
                        except (FileNotFoundError,ValueError): moves=0
                        block_ready=True
                        if execution['mode']=='act_temporal':
                            try:
                                rows=[json.loads(s) for s in (path/'executor.jsonl').read_text().splitlines()]
                                # Stop inside a multi-step commitment, exercising
                                # cancellation of the remaining locked targets.
                                threshold=305 if execution['n_action_steps']>1 else 300
                                block_ready=sum(r['event']=='temporal_step_applied' for r in rows)>=threshold
                            except (FileNotFoundError,json.JSONDecodeError): block_ready=False
                        if execution['mode'] in ('chunked','pipelined','chunked_prefetch'):
                            try:
                                rows=[json.loads(s) for s in (path/'executor.jsonl').read_text().splitlines()]
                                block_ready=(sum(r['event']=='chunk_completed' for r in rows)>=2 and
                                             sum(r['event']=='chunk_started' for r in rows)>=3)
                                if execution['mode'] in ('pipelined','chunked_prefetch'):
                                    predictions=[json.loads(s) for s in (path/'policy_client.jsonl').read_text().splitlines()]
                                    # Stop with the fourth prediction buffered behind segment three.
                                    block_ready=block_ready and sum(r['event']=='raw_policy_chunk' for r in predictions)>=4
                            except (FileNotFoundError,json.JSONDecodeError): block_ready=False
                        if moves>=100 and (path/'fake_motion_exercised').exists() and block_ready:break
                        time.sleep(.01)
                    else:raise TimeoutError('fake graph did not move before stop exercise')
                    write_json(path/'stop.request',dict(monotonic_ns=time.monotonic_ns(),reason='software stop exercise'))
                    time.sleep(.15)
                    write_json(path/'disable.request',dict(monotonic_ns=time.monotonic_ns(),reason='software disable exercise'))
                except Exception as exc:stop_errors.append(str(exc))
            stopper=threading.Thread(target=stop_fake_graph,daemon=True) if args.exercise_stop else None
            if stopper:stopper.start()
            with (path/'dora_stdout.txt').open('w') as flow_out:
                completed=subprocess.run([str(ROOT/'.venv/bin/dora'),'run',str(graph_path),'--stop-after',
                                          '24s' if execution['mode']=='act_temporal' else '8s' if args.exercise_stop else '5s'],
                    cwd=ROOT,env=env,stdout=flow_out,stderr=subprocess.STDOUT,timeout=35)
            if completed.returncode:raise RuntimeError((path/'dora_stdout.txt').read_text())
            if stopper:
                stopper.join(timeout=1)
                if stopper.is_alive():raise AssertionError('stop exercise did not finish')
            if stop_errors:raise AssertionError(stop_errors)
            moves=int((path/'fake_moves.txt').read_text())
            assert moves>10
            chunks=sum(1 for line in (path/'policy_client.jsonl').read_text().splitlines()
                       if json.loads(line)['event']=='raw_policy_chunk')
            assert chunks>2
            block_checks={}
            if execution['mode']=='act_temporal':
                import numpy as np
                records=[json.loads(s) for s in (path/'executor.jsonl').read_text().splitlines()]
                predictions=[json.loads(s) for s in (path/'policy_client.jsonl').read_text().splitlines()]
                applied=[r for r in records if r['event']=='temporal_step_applied']
                raw=[r for r in predictions if r['event']=='raw_policy_chunk']
                late=[r for r in predictions if r['event']=='temporal_prediction_late']
                assert len(applied)>=300
                assert all(b['temporal_step']>a['temporal_step'] for a,b in zip(applied,applied[1:]))
                assert all(0<=r['applied_ns']-r['target_ns']<33_333_333 for r in applied)
                assert max(r['contributors'] for r in applied)>=2
                outputs=[r for r in records if r['event']=='executor_output']
                assert all(r['metadata']['n_action_steps']==execution['n_action_steps'] for r in outputs)
                if args.exercise_stop:
                    stop_ns=json.loads((path/'stop.request').read_text())['monotonic_ns']
                    # Physical path also checks the request independently; this
                    # tests that no already-buffered prediction restarts motion.
                    assert all(r['monotonic_ns']<stop_ns+4_000_000 for r in outputs)
                    assert all(r['applied_ns']<stop_ns+4_000_000 for r in applied)
                latencies=[r['round_trip_ms'] for r in raw]+[r['latency_ms'] for r in late]
                block_checks=dict(applied_steps=len(applied),
                    max_contributors=max(r['contributors'] for r in applied),
                    effective_update_hz=(len(applied)-1)*1e9/(applied[-1]['target_ns']-applied[0]['target_ns']),
                    round_trip_ms_percentiles=dict(zip(('p50','p95','max'),np.percentile(latencies,[50,95,100]).tolist())),
                    late_predictions=len(late),
                    transport_late=sum(r['event']=='temporal_transport_late' for r in records),
                    ensemble_resets=sum(r['event']=='temporal_ensemble_reset' for r in records),
                    wait_holds=sum(r['event']=='temporal_wait_hold' for r in records),
                    output_ticks_skipped=sum(r['count'] for r in records if r['event']=='temporal_output_ticks_skipped'),
                    output_count=len(outputs), same_time_alignment=True)
                if execution['n_action_steps']>1:
                    steps=execution['n_action_steps']
                    blocks=[r for r in records if r['event']=='temporal_block_committed']
                    assert len(blocks)>=300//steps
                    assert all(b['target_ns']>=a['expires_ns'] for a,b in zip(blocks,blocks[1:]))
                    complete=0
                    for block in blocks:
                        block_moves=[r for r in applied if r['committed_source_step']==block['temporal_step']]
                        indices=[r['temporal_step']-block['temporal_step'] for r in block_moves]
                        assert indices==list(range(len(indices)))
                        for index,r in zip(indices,block_moves):
                            np.testing.assert_array_equal(r['fused_target'],block['targets'][index])
                            assert r['target_ns']==block['target_ns']+index*33_333_333
                        if len(indices)==steps:complete+=1
                    assert complete>=300//steps-1
                    if args.exercise_stop:
                        assert blocks[-1]['target_ns']<stop_ns<blocks[-1]['expires_ns']
                    block_checks.update(committed_blocks=len(blocks),complete_locked_blocks=complete,
                        locked_steps=steps,
                        frozen_targets_not_overwritten=True,stop_cancels_remaining_targets=args.exercise_stop)
                if args.inference_delay_ms:
                    assert late and any(r['event']=='temporal_wait_hold' for r in records)
                    assert block_checks['ensemble_resets']>=2
                    block_checks['late_prediction_discarded_and_recovered']=True
            if execution['mode']=='chunked':
                records=[json.loads(s) for s in (path/'executor.jsonl').read_text().splitlines()]
                completed=[r for r in records if r['event']=='chunk_completed']
                started=[r for r in records if r['event']=='chunk_started']
                assert len(completed)>=2
                assert all(r['n_action_steps']==execution['n_action_steps'] for r in started)
                assert all(r['elapsed_ns']>=execution['n_action_steps']*33_333_333 for r in completed)
                predictions=[json.loads(s) for s in (path/'policy_client.jsonl').read_text().splitlines()]
                for previous,next_chunk in zip(completed,predictions[1:]):
                    assert next_chunk['metadata']['observation_monotonic_ns']>previous['completed_monotonic_ns']
                    assert (next_chunk['metadata']['observation_monotonic_ns']-next_chunk['metadata']['source_age_ns']
                            >previous['completed_monotonic_ns'])
                block_checks=dict(completed_chunks=len(completed),started_chunks=len(started),
                                  fresh_observation_after_completion=True,
                                  all_sources_after_completion=True)
            elif execution['mode'] in ('pipelined','chunked_prefetch'):
                records=[json.loads(s) for s in (path/'executor.jsonl').read_text().splitlines()]
                completed=[r for r in records if r['event']=='chunk_completed']
                started=[r for r in records if r['event']=='chunk_started']
                credits=[r for r in records if r['event']=='chunk_prefetch_requested']
                predictions=[json.loads(s) for s in (path/'policy_client.jsonl').read_text().splitlines()
                             if json.loads(s)['event']=='raw_policy_chunk']
                assert len(completed)>=2 and len(started)>=3
                assert len(predictions)<=len(credits)+1
                assert started[0]['skipped_prefix_ns']==0
                if execution['mode']=='pipelined':
                    assert all(r['skipped_prefix_ns']>0 for r in started[1:])
                else:
                    assert all(r['skipped_prefix_ns']==0 and r['first_coarse_step']==0 for r in started)
                    assert all(r['elapsed_ns']>=50*33_333_333 for r in completed)
                assert all(r['duration_ns']+r['skipped_prefix_ns']==50*33_333_333 for r in started)
                gaps=[]
                for i,previous in enumerate(completed):
                    if i+1>=len(started):break
                    following=started[i+1]
                    gaps.append((following['execution_started_monotonic_ns']-previous['completed_monotonic_ns'])/1e6)
                    assert following['execution_started_monotonic_ns']>=previous['completed_monotonic_ns']
                    if not args.inference_delay_ms:
                        assert predictions[i+1]['monotonic_ns']<previous['completed_monotonic_ns']
                for c,p in zip(credits,predictions[1:]):
                    assert p['metadata']['observation_monotonic_ns']-p['metadata']['source_age_ns']>c['requested_monotonic_ns']
                for c in completed:
                    segment_moves=[r for r in records if r['event']=='executor_output' and r['output']=='move_position_right'
                           and r['metadata']['chunk_id']==c['chunk_id']]
                    assert segment_moves[-1]['metadata']['coarse_step']==49
                    if execution['mode']=='chunked_prefetch':
                        assert len(segment_moves)==417
                        assert {r['metadata']['coarse_step'] for r in segment_moves}==set(range(50))
                if args.exercise_stop:
                    stop_ns=json.loads((path/'stop.request').read_text())['monotonic_ns']
                    assert all(r['execution_started_monotonic_ns']<stop_ns for r in started)
                    assert len(predictions)>len(started), 'STOP must exercise a buffered prediction'
                block_checks=dict(completed_chunks=len(completed),started_chunks=len(started),
                    prefetched_chunks=len(predictions)-1,inter_chunk_gap_ms=gaps,
                    prefetched_before_completion=not bool(args.inference_delay_ms),stop_with_buffered_prediction=args.exercise_stop,
                    all_completed_segments_reach_step49=True)
                if execution['mode']=='chunked_prefetch':
                    block_checks['all_completed_segments_execute_full50']=True
                    block_checks['observation_age_at_start_ms']=[r['observation_age_ns']/1e6 for r in started]
                first_last=[]
                for s in started:
                    segment_moves=[r for r in records if r['event']=='executor_output' and r['output']=='move_position_right'
                                   and r['metadata']['chunk_id']==s['chunk_id']]
                    first_last.append((segment_moves[0]['monotonic_ns'],segment_moves[-1]['monotonic_ns']))
                block_checks['boundary_output_interval_ms']=[(b[0]-a[1])/1e6 for a,b in zip(first_last,first_last[1:])]
                if args.inference_delay_ms:
                    assert gaps[0]>args.inference_delay_ms-execution['prefetch_ms']
                    if execution['mode']=='pipelined':
                        block_checks['late_result_aligned']=started[1]['skipped_prefix_ns']>args.inference_delay_ms*1e6
                    else:
                        assert started[1]['skipped_prefix_ns']==0
                        assert started[1]['observation_age_ns']>args.inference_delay_ms*1e6
                        block_checks['late_result_retains_full50']=True
            if args.collection_feedforward:
                faults=list(path.glob('*_fault.json'))
                if faults:raise AssertionError([f.read_text() for f in faults])
                commands=[json.loads(line) for line in (path/'feedforward_right.jsonl').read_text().splitlines()]
                assert any(any(v!=0 for v in c['dq_ref']) for c in commands if c['event']=='mit_command')
                assert any(any(v!=0 for v in c['tau_ff']) for c in commands if c['event']=='mit_command')
                if args.inference_delay_ms and execution['mode']!='act_temporal':
                    between=[c for c in commands if c['event']=='mit_command' and
                             completed[0]['completed_monotonic_ns']<c['monotonic_ns']<started[1]['execution_started_monotonic_ns']]
                    assert between and all(all(v==0 for v in c['dq_ref']) for c in between)
                    assert all(c['q_ref']==between[0]['q_ref'] for c in between)
                    block_checks['late_wait_holds_position_zero_velocity']=True
                if args.exercise_stop:
                    for side in ('right','left'):
                        disabled=json.loads((path/f'fake_disabled_{side}.json').read_text())
                        records=[json.loads(x) for x in (path/f'feedforward_{side}.jsonl').read_text().splitlines()]
                        held=[c for c in records if c['event']=='mit_command' and c['held']]
                        assert held and all(all(v==0 for v in c['dq_ref']) for c in held)
                        assert all(c['monotonic_ns']<=disabled['monotonic_ns'] for c in records)
            report=dict(passed=True,hardware_accessed=False,physical_validation_passed=False,
                        fake_motor_targets=moves,real_policy_chunks=chunks,logs=str(path),
                        collection_feedforward=args.collection_feedforward,stop_disable_exercised=args.exercise_stop)
            report.update(action_execution=execution,block_checks=block_checks)
            report['saved_images']=args.saved_images
            report['test_inference_delay_ms']=args.inference_delay_ms
            (ROOT/'validation').mkdir(exist_ok=True)
            write_json(args.report or ROOT/'validation/dataflow_smoke.json',report);print(json.dumps(report,indent=2))
        finally:
            if policy.poll() is None:policy.terminate()
            try:policy.wait(timeout=5)
            except subprocess.TimeoutExpired:policy.kill();policy.wait()


if __name__=='__main__':main()
