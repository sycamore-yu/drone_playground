"""Source-only distributions must preserve provenance and subsequent local edits."""

import json
import subprocess
import tarfile

import pytest

from drone_playground.artifacts import RunRecorder
from drone_playground.artifacts.source_snapshot import export_source


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


@pytest.fixture
def exported(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    (repo / '.gitignore').write_text('results/\n__pycache__/\n.env\n')
    (repo / 'source.py').write_text('VALUE = 1\n')
    (repo / 'removed.py').write_text('OLD = True\n')
    (repo / 'alias.py').symlink_to('source.py')
    git(repo, 'add', '.')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'source fixture')
    revision = git(repo, 'rev-parse', 'HEAD')
    (repo / '.env').write_text('NOT_FOR_EXPORT=secret\n')
    (repo / 'untracked.txt').write_text('not in release\n')
    archive = tmp_path / 'source.tar.gz'
    result = export_source(repo, archive)
    assert result['source_revision'] == revision
    destination = tmp_path / 'unpacked'
    destination.mkdir()
    with tarfile.open(archive) as handle:
        assert '.env' not in handle.getnames() and 'untracked.txt' not in handle.getnames()
        handle.extractall(destination, filter='data')
    return repo, archive, destination, revision


def record(root, name):
    recorder = RunRecorder(root, name, {'scope': 'source evidence only'})
    recorder.finish('completed')
    return recorder.path, json.loads((recorder.path / 'manifest.json').read_text())


def test_export_is_deterministic_and_gitless_run_has_verified_origin(exported):
    repo, archive, root, revision = exported
    second = archive.with_name('another-name.tar.gz')
    export_source(repo, second)
    assert archive.read_bytes() == second.read_bytes()
    path, manifest = record(root, 'pristine')
    assert manifest['code']['commit'] is None
    evidence = manifest['code']['archive']
    assert evidence['source_revision'] == revision and evidence['dirty'] is False
    assert (path / evidence['snapshot_path']).is_file()
    # Earlier output artifacts must not turn the source dirty on the next run.
    _, later = record(root, 'second')
    assert later['code']['archive']['dirty'] is False


def test_changed_deleted_and_new_sources_are_recorded_in_actual_snapshot(exported):
    _, _, root, _ = exported
    (root / 'source.py').write_text('VALUE = 7\n')
    (root / 'removed.py').unlink()
    (root / 'new.py').write_text('NEW = 3\n')
    (root / '.env').write_text('LOCAL_SECRET=not_source\n')
    path, manifest = record(root, 'edited')
    evidence = manifest['code']['archive']
    assert evidence['dirty'] is True
    assert evidence['modified'] == ['source.py']
    assert evidence['removed'] == ['removed.py']
    assert evidence['added'] == ['new.py']
    with tarfile.open(path / evidence['snapshot_path']) as handle:
        assert handle.extractfile('source.py').read() == b'VALUE = 7\n'
        assert handle.extractfile('new.py').read() == b'NEW = 3\n'
        assert 'removed.py' not in handle.getnames() and '.env' not in handle.getnames()
        assert handle.getmember('alias.py').issym()


def test_extracted_source_does_not_inherit_an_unrelated_parent_commit(exported):
    repo, archive, _, revision = exported
    root = repo / 'nested'
    root.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(root, filter='data')
    _, manifest = record(root, 'nested')
    assert manifest['code']['commit'] is None
    assert manifest['code']['archive']['source_revision'] == revision
    assert manifest['code']['archive']['dirty'] is False


def test_invalid_manifest_path_is_rejected(exported):
    _, _, root, _ = exported
    path = root / 'SOURCE_MANIFEST.json'
    manifest = json.loads(path.read_text())
    manifest['files']['../outside.py'] = manifest['files']['source.py']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r'source manifest path'):
        record(root, 'invalid')


def test_symlink_changes_do_not_copy_external_content(exported, tmp_path):
    _, _, root, _ = exported
    outside = tmp_path / 'outside.py'
    outside.write_text('PRIVATE_CONTENT = True\n')
    (root / 'alias.py').unlink()
    (root / 'alias.py').symlink_to(outside)
    (root / 'source.py').chmod(0o755)
    path, manifest = record(root, 'symlink-edit')
    evidence = manifest['code']['archive']
    assert evidence['modified'] == ['alias.py', 'source.py']
    with tarfile.open(path / evidence['snapshot_path']) as handle:
        assert handle.getmember('alias.py').issym()
        assert handle.getmember('source.py').mode == 0o755
        assert all(b'PRIVATE_CONTENT' not in handle.extractfile(item).read()
                   for item in handle if item.isfile())
