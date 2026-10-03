"""Fresh and reused dependency caches must validate added patch files without staging."""

import importlib.util
import json
import subprocess

import pytest

from tests.helpers.paths import REPO_ROOT

SCRIPT = REPO_ROOT / 'scripts/tools/setup.py'
SPEC = importlib.util.spec_from_file_location('setup_tool', SCRIPT)
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
    patches = root / 'third_party'
    patches.mkdir(parents=True)
    patch = SCRIPT.parents[2] / 'third_party/patches/lotf-package-metadata.patch'
    # This isolated fixture exercises added packaging files, without copying the
    # physical dependency. Its model patch is covered against the real source.
    (patches / 'metadata.patch').write_text(patch.read_text().split('diff --git a/lotf/objects/')[0])
    (patches / 'sources.json').write_text(json.dumps({'lotf': {
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
    with pytest.raises(ValueError, match=r'Cached modifications|Unexpected cached'):
        MODULE.fetch(source_project)
    assert target.read_text() == 'local user edit\n'
    assert (cache / '.git/index').read_bytes() == index
