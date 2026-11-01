"""Translate operator requests to official node messages; never access CAN."""
import dora
import pyarrow as pa
from .common import session, Log


def run(node, *, hardware=False, log=None):
    previous=None
    enable_sent=False
    disable_sent=False
    hold_sent=False
    for event in node:
        if event['type']!='INPUT': continue
        path=session()
        stopped=(path/'stop.request').exists() or (path/'disable.request').exists()
        active=(path/'start.request').exists() and not stopped
        if hardware:
            if (path/'disable.request').exists() and not disable_sent:
                node.send_output('arm_command',pa.array(['stop']))
                disable_sent=True
                if log: log('official_stop_requested',disable_confirmed=False)
            elif stopped and enable_sent and not hold_sent:
                node.send_output('arm_command',pa.array(['hold']))
                hold_sent=True
                if log: log('feedforward_hold_requested',reference_velocity=0)
            elif not stopped and not enable_sent and any(
                    (path/f'{name}.request').exists() for name in ('enable','home')):
                node.send_output('arm_command',pa.array(['start']))
                enable_sent=True
                if log: log('official_start_requested')
        state='start' if active else 'stop'
        if state!=previous:
            node.send_output('command',pa.array([state]),{'episode_number':1})
            node.send_output('task_prompt',pa.array(['pick up cube']))
            previous=state
            if log: log('policy_routing',active=active)
        if hardware and active and event['id']=='move_position':
            # Forward the official executor output unchanged, no interpolation.
            node.send_output('move_position_right',event['value'],event['metadata'])


def main(hardware=False):
    run(dora.Node(),hardware=hardware,log=Log('operator') if hardware else None)
