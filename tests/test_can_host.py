"""Cold-host regression tests: fake procfs and commands, no CAN/cameras."""
from contextlib import nullcontext
from unittest.mock import Mock
import json
import sys

import pytest

from deployment import can_ownership, host_setup, cli


def registry(tmp_path):
    (tmp_path/'rcvlist_all').write_text('device can_id can_mask function userdata matches ident\n')
    return tmp_path


def link(up=True, mtu=72):
    return dict(ifname='can0', link_type='can', flags=['UP'] if up else [], mtu=mtu,
                operstate='UNKNOWN', linkinfo={'info_data': {'state': 'ERROR-ACTIVE'}})


def test_missing_registry_actionable_and_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match='sudo modprobe can_raw'):
        can_ownership.reject_existing_receivers('can0', tmp_path/'missing')
    with pytest.raises(RuntimeError, match='registry unavailable'):
        can_ownership.reject_existing_receivers('can0', tmp_path)


@pytest.mark.parametrize('device', ['can0', '(any)', 'any'])
def test_existing_receivers_still_rejected(tmp_path, device):
    registry(tmp_path)
    (tmp_path/'rcvlist_all').write_text(f'{device} 000 000 receiver\n')
    with pytest.raises(RuntimeError, match='Existing CAN receiver'):
        can_ownership.reject_existing_receivers('can0', tmp_path)


def test_lock_released_on_receiver_failure_and_persistent(tmp_path):
    registry(tmp_path)
    (tmp_path/'rcvlist_all').write_text('can0 000 000 receiver\n')
    with pytest.raises(RuntimeError, match='Existing CAN receiver'):
        can_ownership.CanOwnership('can0', 'test', _test_root=tmp_path)
    (tmp_path/'rcvlist_all').write_text('')
    with can_ownership.CanOwnership('can0', 'test', _test_root=tmp_path):
        with pytest.raises(RuntimeError, match='already owned'):
            can_ownership.CanOwnership('can0', 'test2', _test_root=tmp_path)
    assert (tmp_path/'can0.lock').exists()


def test_readonly_host_check_down_fd_and_unknown_operstate(monkeypatch, tmp_path):
    read=Mock(return_value=json.dumps([link(False, 16)]))
    monkeypatch.setattr(host_setup.subprocess, 'check_output', read)
    monkeypatch.setattr(host_setup.subprocess, 'run', Mock(side_effect=AssertionError('no mutations')))
    with pytest.raises(RuntimeError, match='DOWN.*CAN-FD'):
        host_setup.require_can_host(['can0'], registry=registry(tmp_path))
    read.return_value=json.dumps([link()])
    host_setup.require_can_host(['can0'], registry=tmp_path)
    read.return_value='[]'
    with pytest.raises(RuntimeError, match='missing interface'):
        host_setup.require_can_host(['can0'], registry=tmp_path)


def setup_mocks(monkeypatch, links):
    run=Mock()
    monkeypatch.setattr(host_setup.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(host_setup.subprocess, 'run', run)
    monkeypatch.setattr(host_setup, 'receiver_files', Mock())
    monkeypatch.setattr(host_setup, 'CanOwnership', lambda *args: nullcontext())
    monkeypatch.setattr(host_setup, 'read_links', lambda: links)
    monkeypatch.setattr(host_setup, 'require_can_host', Mock())
    return run


def test_explicit_setup_only_configures_down_links(monkeypatch):
    run=setup_mocks(monkeypatch, {'can0': link(False,16), 'can1': link()})
    host_setup.prepare_can_host(['can0','can1'])
    commands=[call.args[0] for call in run.call_args_list]
    assert commands == [
        ['sudo','modprobe','can_raw'],
        ['sudo','ip','link','set','dev','can0','type','can','bitrate','1000000',
         'sample-point','0.75','dbitrate','5000000','dsample-point','0.75','dsjw','2','fd','on'],
        ['sudo','ip','link','set','dev','can0','up']]


def test_setup_checks_both_owners_before_link_changes(monkeypatch):
    run=setup_mocks(monkeypatch, {'can0': link(False,16), 'can1': link(False,16)})
    def owner(interface, backend):
        if interface=='can1':
            raise RuntimeError('already owned')
        return nullcontext()
    monkeypatch.setattr(host_setup, 'CanOwnership', owner)
    with pytest.raises(RuntimeError, match='already owned'):
        host_setup.prepare_can_host(['can0','can1'])
    assert [c.args[0] for c in run.call_args_list] == [['sudo','modprobe','can_raw']]


def test_dora_preflight_fails_before_marker_or_exec(monkeypatch, tmp_path):
    (tmp_path/'driver.yaml').write_text('can_interface: {right_arm: can0, left_arm: can1}\n')
    monkeypatch.setattr(sys,'argv',['actctl','dora'])
    monkeypatch.setattr(cli,'load_session',lambda: tmp_path)
    monkeypatch.setattr(cli,'verify_sources',lambda: None)
    monkeypatch.setattr(cli,'assert_unchanged',lambda p: None)
    monkeypatch.setattr(host_setup,'require_can_host',Mock(side_effect=RuntimeError('registry unavailable')))
    monkeypatch.setattr(cli.os,'execvpe',Mock(side_effect=AssertionError('no dataflow')))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert not (tmp_path/'dora_process.json').exists()


def test_recovery_downs_both_before_up_preserves_config_and_records(monkeypatch):
    run=setup_mocks(monkeypatch, {'can0':link(), 'can1':link()})
    events=[]
    host_setup.recover_can_host(['can0','can1'],record=lambda event,**fields:events.append((event,fields)))
    commands=[c.args[0] for c in run.call_args_list]
    assert commands==[['sudo','ip','link','set','dev',i,s]
                      for s in ('down','up') for i in ('can0','can1')]
    assert events[0][0]=='before' and events[-1][0]=='after'
    assert sum(e=='command_completed' for e,_ in events)==4


def test_recovery_busy_second_interface_changes_neither(monkeypatch):
    run=setup_mocks(monkeypatch, {'can0':link(), 'can1':link()})
    def owner(interface,backend):
        if interface=='can1':raise RuntimeError('already owned')
        return nullcontext()
    monkeypatch.setattr(host_setup,'CanOwnership',owner)
    with pytest.raises(RuntimeError,match='already owned'):
        host_setup.recover_can_host(['can0','can1'])
    run.assert_not_called()


def test_recovery_down_failure_never_raises_either_link(monkeypatch):
    import subprocess
    run=setup_mocks(monkeypatch, {'can0':link(), 'can1':link()})
    run.side_effect=[None,subprocess.CalledProcessError(1,['ip'])]
    with pytest.raises(subprocess.CalledProcessError):
        host_setup.recover_can_host(['can0','can1'])
    assert all(c.args[0][-1]=='down' for c in run.call_args_list)
    assert run.call_count==2
