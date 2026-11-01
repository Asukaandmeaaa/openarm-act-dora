"""Release installation rejects wrong weights and archive links, without hardware."""
import hashlib
import io
import tarfile
import pytest
from scripts.fetch_model import install, sha256, verify


def bundle(tmp_path, name='config.json', kind=tarfile.REGTYPE):
    archive=tmp_path/'bundle.tar.gz'
    data=b'{"type":"act"}'
    with tarfile.open(archive,'w:gz') as tar:
        member=tarfile.TarInfo(name);member.type=kind
        if kind==tarfile.REGTYPE:member.size=len(data)
        else:member.linkname='/tmp/should-not-be-written'
        tar.addfile(member,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
    manifest={'archive_sha256':sha256(archive),'files':{'config.json':hashlib.sha256(data).hexdigest()}}
    return archive,manifest


def test_download_install_verify_and_preserve_existing(tmp_path):
    archive,manifest=bundle(tmp_path)
    dest=tmp_path/'models/checkpoint'
    install(archive,dest,manifest);verify(dest,manifest)
    with pytest.raises(FileExistsError):install(archive,dest,manifest)
    (dest/'config.json').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):verify(dest,manifest)


def test_bad_archive_hash_never_creates_destination(tmp_path):
    archive,manifest=bundle(tmp_path);manifest['archive_sha256']='0'*64
    dest=tmp_path/'models/checkpoint'
    with pytest.raises(ValueError,match='SHA256'):install(archive,dest,manifest)
    assert not dest.exists()


@pytest.mark.parametrize('name,kind',[('../config.json',tarfile.REGTYPE),('config.json',tarfile.SYMTYPE)])
def test_archive_path_traversal_and_links_rejected(tmp_path,name,kind):
    archive,manifest=bundle(tmp_path,name,kind);dest=tmp_path/'models/checkpoint'
    with pytest.raises(ValueError):install(archive,dest,manifest)
    assert not dest.exists()
