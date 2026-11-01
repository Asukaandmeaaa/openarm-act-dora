"""Local-only Git fixture proves clean install and interrupted-fetch recovery."""
import json
import subprocess
import sys
import pytest
from scripts import bootstrap_sources as bootstrap


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


@pytest.mark.parametrize('interrupted',[False,True])
def test_fetch_apply_idempotent_and_reject_dirty(tmp_path,monkeypatch,interrupted):
    source=tmp_path/'origin';source.mkdir()
    git(source,'init','-q');git(source,'config','user.name','test');git(source,'config','user.email','test@example.invalid')
    (source/'main.py').write_text('x = 1\n\n')
    git(source,'add','main.py');git(source,'commit','-qm','base');commit=git(source,'rev-parse','HEAD')
    (source/'main.py').write_text('x = 2\n\n')
    patch=subprocess.check_output(['git','-C',str(source),'diff','--binary','HEAD'])
    root=tmp_path/'release';root.mkdir();(root/'change.patch').write_bytes(patch)
    dest=root/'upstream/test'
    (root/'sources.lock.json').write_text(json.dumps({'repositories':[{'path':'upstream/test','url':str(source),'commit':commit,'patch':'change.patch'}]}))
    if interrupted:
        dest.mkdir(parents=True);git(dest,'init','-q');git(dest,'remote','add','origin',str(source))
    monkeypatch.setattr(bootstrap,'ROOT',root);monkeypatch.setattr(sys,'argv',['bootstrap_sources.py'])
    bootstrap.main();bootstrap.main()
    assert (dest/'main.py').read_text()=='x = 2\n\n'
    (dest/'main.py').write_text('x = 3\n')
    with pytest.raises(RuntimeError,match='Unrecorded'):bootstrap.main()
