"""Fresh and reused dependency caches must validate added patch files without staging."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/tools/fetch_sources.py'
SPEC = importlib.util.spec_from_file_location('fetch_sources', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def git(directory, *args):
    return subprocess.check_output(['git', *args], cwd=directory, text=True).strip()


@pytest.fixture
def source_project(tmp_path):
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    git(upstream, 'init', '-q')
    (upstream / 'original.txt').write_text('upstream\n')
    git(upstream, 'add', 'original.txt')
    git(upstream, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'fixture')
    revision = git(upstream, 'rev-parse', 'HEAD')
    root = tmp_path / 'project'
    third_party = root / 'third_party'
    third_party.mkdir(parents=True)
    patch = SCRIPT.parents[2] / 'third_party/patches/lotf-package-metadata.patch'
    (third_party / 'metadata.patch').write_bytes(patch.read_bytes())
    (third_party / 'sources.yaml').write_text(json.dumps({'lotf': {
        'repository': str(upstream), 'revision': revision,
        'checkout': 'tmp/sources/lotf', 'patch': 'third_party/metadata.patch',
    }}))
    return root


def test_added_metadata_patch_validates_fresh_and_reused_without_staging(source_project):
    first = MODULE.fetch(source_project)
    cache = source_project / first['lotf']['path']
    index = (cache / '.git/index').read_bytes()
    assert (cache / 'pyproject.toml').is_file()
    assert git(cache, 'ls-files') == 'original.txt'
    assert MODULE.fetch(source_project) == first
    assert (cache / '.git/index').read_bytes() == index


@pytest.mark.parametrize('filename', ['pyproject.toml', 'original.txt', 'unexpected.py'])
def test_changed_dependency_is_rejected_and_never_overwritten(source_project, filename):
    MODULE.fetch(source_project)
    cache = source_project / 'tmp/sources/lotf'
    target = cache / filename
    target.write_text('local user edit\n')
    index = (cache / '.git/index').read_bytes()
    with pytest.raises(ValueError, match='Cached modifications|Unexpected cached'):
        MODULE.fetch(source_project)
    assert target.read_text() == 'local user edit\n'
    assert (cache / '.git/index').read_bytes() == index
