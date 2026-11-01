"""Host setup is operator-only; diagnostics never modify interfaces.

Neither path opens CAN sockets or sends motor commands. Link configuration is
the documented configuration of this workstation's gs_usb adapters.
"""
from contextlib import ExitStack
import json
import os
import subprocess

from .can_ownership import CanOwnership, reject_existing_receivers, receiver_files


def read_links():
    try:
        links = json.loads(subprocess.check_output(
            ['ip', '-details', '-json', 'link', 'show'], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f'Cannot inspect CAN interfaces with ip: {exc}') from exc
    return {link['ifname']: link for link in links}


def require_can_host(interfaces, *, registry='/proc/net/can'):
    interfaces = tuple(dict.fromkeys(interfaces))
    # Check before spawning any nodes. Robot nodes still acquire their own locks
    # and recheck registrations immediately before constructing the SDK.
    for interface in interfaces:
        reject_existing_receivers(interface, registry)
    by_name = read_links()
    errors = []
    for interface in interfaces:
        link = by_name.get(interface)
        if link is None:
            errors.append(f'{interface}: missing interface')
        elif link.get('link_type') != 'can':
            errors.append(f'{interface}: not a CAN interface')
        else:
            if 'UP' not in link.get('flags', []):
                errors.append(f'{interface}: DOWN')
            if link.get('mtu') != 72:
                errors.append(f'{interface}: CAN-FD is not enabled (MTU={link.get("mtu")})')
            if link.get('linkinfo', {}).get('info_data', {}).get('state') == 'BUS-OFF':
                errors.append(f'{interface}: BUS-OFF')
    if errors:
        raise RuntimeError('CAN host is not ready: ' + '; '.join(errors) +
                           '. Run ./scripts/actctl prepare-host before starting Dora; see docs/HARDWARE.md. '
                           'No interfaces were changed and no nodes were started.')


def prepare_can_host(interfaces):
    """Explicit operator command only; never called from check/prepare/dora."""
    interfaces = tuple(dict.fromkeys(interfaces))
    prefix = [] if os.geteuid() == 0 else ['sudo']
    subprocess.run([*prefix, 'modprobe', 'can_raw'], check=True)
    receiver_files('/proc/net/can')
    with ExitStack() as stack:
        # Acquire both before touching either; also catches unpatched receivers.
        for interface in interfaces:
            stack.enter_context(CanOwnership(interface, 'dora-host-setup'))
        links = read_links()
        for interface in interfaces:
            if interface not in links or links[interface].get('link_type') != 'can':
                raise RuntimeError(f'{interface}: missing CAN interface; check USB adapter/configuration')
        for interface in interfaces:
            if 'UP' in links[interface].get('flags', []):
                print(f'{interface}: already UP; leaving configuration unchanged')
                continue
            subprocess.run([*prefix, 'ip', 'link', 'set', 'dev', interface, 'type', 'can',
                            'bitrate', '1000000', 'sample-point', '0.75',
                            'dbitrate', '5000000', 'dsample-point', '0.75',
                            'dsjw', '2', 'fd', 'on'], check=True)
            subprocess.run([*prefix, 'ip', 'link', 'set', 'dev', interface, 'up'], check=True)
        require_can_host(interfaces)
    print('CAN host ready. Existing UP links were preserved; no motor commands sent. '
          'ENABLE/START remain operator actions.')


def recover_can_host(interfaces, *, record=lambda event, **fields: None):
    """Operator-only down/up recovery. Never called by hardware/prepare/check.

    The operator must first support/power off the arms: bringing a link back up
    must not allow old queued targets to move an enabled arm. All interfaces are
    brought DOWN before any is raised. Existing bitrates/FD settings are kept.
    """
    interfaces=tuple(dict.fromkeys(interfaces))
    prefix=[] if os.geteuid()==0 else ['sudo']
    with ExitStack() as stack:
        for interface in interfaces:
            stack.enter_context(CanOwnership(interface,'wave-host-recovery'))
        links=read_links()
        for interface in interfaces:
            link=links.get(interface,{})
            if link.get('link_type')!='can' or link.get('mtu')!=72:
                raise RuntimeError(f'{interface}: expected existing CAN-FD interface; recovery cancelled')
        record('before',interfaces={i:links[i] for i in interfaces})
        for state in ('down','up'):
            for interface in interfaces:
                command=[*prefix,'ip','link','set','dev',interface,state]
                record('command_attempt',command=command)
                subprocess.run(command,check=True)
                record('command_completed',command=command)
        require_can_host(interfaces)
        record('after',interfaces={i:link for i,link in read_links().items() if i in interfaces})
    print('CAN interfaces cycled down/up. No motor enable/start commands sent. '
          'This does not confirm motor communication or a physical disable.')
