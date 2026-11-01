"""Acquire CAN ownership, then enter the pinned official node with listed patches.

The official node constructs SingleArmDriver only on an operator start event.
Optional collection feedforward uses that same driver and CAN writer.
"""
import sys
from .can_ownership import CanOwnership
from .common import session, request, write_json


def main(side):
    import openarm_driver
    from dora_openarm import main as official
    path=session()/'driver.yaml'
    config=openarm_driver.Config(path)
    sys.argv=['dora-openarm', '--side', side, '--config', str(path),
              '--no-align', '--no-start-on-startup', '--stop']
    try:
        with CanOwnership(config.get_can_interface(side+'_arm'), 'official-dora-native'):
            official.main()
    except BaseException as exc:
        reason=f'{type(exc).__name__}: {exc}'
        request('stop', reason)
        fault_path=session()/f'{side}_fault.json'
        if not fault_path.exists(): write_json(fault_path, {'reason':reason})
        raise
