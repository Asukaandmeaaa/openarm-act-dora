"""Record official node outputs and show their status; never control hardware."""
import json
import time
import dora
from .common import session, Log, write_json


def run(node, logs):
    records={side:dict(enabled=None,active=False,fault=None,stopped=True,
                       official_status='unknown',q=None,qd=None,
                       monotonic_ns=0,feedback_monotonic_ns=0,
                       disable_attempted=False,disable_confirmed=False)
             for side in ('right','left')}
    for event in node:
        if event['type']!='INPUT': continue
        side,kind=event['id'].split('_',1)
        data=event['value'][0].as_py()
        metadata=dict(event.get('metadata') or {})
        now=time.monotonic_ns()
        logs[side](kind,value=data,metadata=metadata)
        record=records[side]
        path=session()
        if kind=='status':
            record['official_status']=data
            # Official 'stopped' precedes completion of stop(); no motor ack.
            record['enabled']=True if data in ('started','aligned') else None
        elif kind=='state':
            record.update(q=data['qpos'],qd=data['qvel'],motor_status=data['motor_status'],
                          bus=data['bus'],feedback_monotonic_ns=now)
        elif kind=='commanded_position':
            # Dora adds a datetime timestamp; JSON snapshots must serialize it.
            record.update(commanded_q=data['qpos'],
                          command_metadata=json.loads(json.dumps(metadata,default=str)))
        stopping=(path/'stop.request').exists() or (path/'disable.request').exists()
        record['stopped']=stopping or record['official_status']=='stopped'
        record['active']=(path/'start.request').exists() and not record['stopped']
        record['disable_attempted']=(path/'disable.request').exists()
        if record['disable_attempted']: record['enabled']=None
        record['monotonic_ns']=now
        write_json(path/f'{side}_status.json',record)


def main():
    run(dora.Node(),{side:Log('official_'+side) for side in ('right','left')})
