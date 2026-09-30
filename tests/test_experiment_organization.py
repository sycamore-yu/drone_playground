import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def organizer():
    path = Path(__file__).parents[1] / 'scripts/tools/organize_experiments.py'
    spec = importlib.util.spec_from_file_location('experiment_organizer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_move_preserves_report_bytes_and_updates_cross_run_links(tmp_path, organizer):
    a, b = [tmp_path / 'experiments' / name for name in ['first-20260929', 'second-20260929']]
    a.mkdir(parents=True)
    b.mkdir()
    payload = b' {"immutable": true}\n'
    (a / 'report.json').write_bytes(payload)
    (b / 'previous').symlink_to(a, target_is_directory=True)
    organizer.move_working_results(tmp_path, apply=False)
    assert (a / 'report.json').read_bytes() == payload
    rows = organizer.move_working_results(tmp_path, apply=True)
    assert len(rows) == 2
    moved = tmp_path / 'experiments/tmp/260929'
    assert (moved / a.name / 'report.json').read_bytes() == payload
    assert (moved / b.name / 'previous').resolve() == moved / a.name


def test_active_run_prevents_all_moves(tmp_path, organizer):
    for name in ['a-completed', 'z-active']:
        (tmp_path / 'experiments' / name).mkdir(parents=True)
    pid = os.getpid()
    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    organizer.write(tmp_path/'experiments/z-active/state.json', dict(status='running', process=dict(pid=pid, start_ticks=ticks)))
    with pytest.raises(RuntimeError, match='active'):
        organizer.move_working_results(tmp_path, apply=True)
    assert (tmp_path/'experiments/a-completed').is_dir()


def test_legacy_list_result_has_an_explicit_date_fallback(tmp_path, organizer):
    run = tmp_path / 'run-20260929'
    run.mkdir()
    (run/'result.json').write_text('[{"seed": 0}]')
    assert organizer.run_date(run) == ('260929', 'directory_name')


def test_best_development_weights_are_copied_without_becoming_a_formal_pass(tmp_path, organizer):
    run = tmp_path / 'experiments/cloud-trial-20260930'
    weights = run / 'training-state/update-0000125.pkl'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'trusted original checkpoint bytes')
    organizer.write(weights.with_suffix('.json'), dict(sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), parameter_sha256='params'))
    organizer.write(run/'resolved-config.json', dict(mode='train', method=dict(trainable=True), training=dict(seed=0)))
    organizer.write(run/'manifest.json', dict(started_at='2026-09-30T00:00:00+00:00', code=dict(commit='frozen')))
    organizer.write(run/'result.json', dict(selected=dict(updates=125)))
    report = dict(quality_passed=False, checkpoint=str(weights), parameter_sha256='params')
    organizer.write(run/'eval/update-0000125.json', report)
    original = (run/'eval/update-0000125.json').read_bytes()
    progress = dict(cells=[dict(method='pointcloud', task='static', status='development_not_confirmed', passed=False)],
                    passed_cells=0, user_accepted_cells=0, release_requirement_satisfied_cells=0,
                    required_cells=1, remaining_quality_cells=1)
    index = organizer.build_main_results(tmp_path, progress, 'test-goal', run.name)
    assert not index[0]['quality_passed'] and not index[0]['release_requirement_satisfied']
    selected = tmp_path/'experiments/main_result/test-goal/01-pointcloud-static/260930-seed0-dev'
    assert (selected/'report.json').read_bytes() == original
    assert (selected/'weights'/weights.name).read_bytes() == weights.read_bytes()
    assert json.loads((selected/'selection.json').read_text())['development_only']
    next_run = run.with_name('cloud-next-trial-20260930')
    shutil.copytree(run, next_run)
    organizer.write(next_run/'eval/update-0000125.json', {**report, 'arrived': 6})
    next_index = organizer.build_main_results(tmp_path, progress, 'test-goal', next_run.name)
    assert (selected/'report.json').read_bytes() == original
    next_item = selected.with_name(f'{selected.name}-{next_run.name}')
    assert json.loads((next_item/'report.json').read_text())['arrived'] == 6
    assert next_index[0]['runs'][0]['run_id'] == next_run.name
