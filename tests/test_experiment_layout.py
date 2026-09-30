from pathlib import Path

import pytest

from drone_playground.runs.layout import (
    experiment_directory,
    find_experiment,
    iter_experiments,
    resolve_artifact,
)


def test_new_runs_use_date_and_existing_runs_keep_their_identity(tmp_path):
    target = experiment_directory(tmp_path, 'trial-a', date='260930')
    assert target == tmp_path / 'experiments/tmp/260930/trial-a'
    target.mkdir(parents=True)
    assert experiment_directory(tmp_path, 'trial-a', date='261001') == target
    assert find_experiment(tmp_path, 'trial-a') == target


def test_legacy_directories_are_readable_and_not_repeated(tmp_path):
    old = tmp_path / 'experiments/legacy-run'
    old.mkdir(parents=True)
    assert experiment_directory(tmp_path, old.name, date='260930') == old


def test_ambiguous_run_identity_is_rejected(tmp_path):
    for date in ['260929', '260930']:
        (tmp_path / 'experiments/tmp' / date / 'duplicate').mkdir(parents=True)
    with pytest.raises(ValueError, match='Ambiguous'):
        find_experiment(tmp_path, 'duplicate')


def test_relocated_checkpoint_is_read_without_rewriting_original_metadata(tmp_path):
    moved = tmp_path / 'experiments/tmp/260929/run-a/training-state/model.pkl'
    moved.parent.mkdir(parents=True)
    moved.write_bytes(b'original checkpoint')
    old = tmp_path / 'experiments/run-a/training-state/model.pkl'
    assert resolve_artifact(old) == moved
    assert resolve_artifact(tmp_path / 'experiments/run-a') == moved.parents[1]
    assert resolve_artifact(moved) == moved
    unrelated = tmp_path / 'some-other-data/missing.pkl'
    assert resolve_artifact(unrelated) == unrelated


def test_listing_does_not_count_final_views_or_date_directories_as_runs(tmp_path):
    for relative in ['experiments/old', 'experiments/tmp/260929/first', 'experiments/tmp/260930/second']:
        (tmp_path / relative).mkdir(parents=True)
    (tmp_path / 'experiments/main_result/v1-18-cells').mkdir(parents=True)
    assert {p.name for p in iter_experiments(tmp_path)} == {'old', 'first', 'second'}


@pytest.mark.parametrize('run_id', ['../escape', '/absolute', '.', '..', 'tmp', 'main_result'])
def test_run_identifiers_cannot_escape_the_layout(tmp_path, run_id):
    with pytest.raises(ValueError):
        experiment_directory(tmp_path, run_id, date='260930')


def test_date_must_be_an_actual_six_digit_calendar_date(tmp_path):
    with pytest.raises(ValueError):
        experiment_directory(tmp_path, 'new-run', date='260931')


def test_same_source_aliases_do_not_make_lookup_ambiguous(tmp_path):
    real = tmp_path / 'experiments/tmp/260929/run-a'
    real.mkdir(parents=True)
    alias = tmp_path / 'experiments/run-a'
    alias.symlink_to(real, target_is_directory=True)
    assert find_experiment(tmp_path, 'run-a').resolve() == real


def test_cli_status_reads_nested_runs_and_legacy_runs(tmp_path, monkeypatch, capsys):
    import json
    from datetime import datetime, timezone

    from drone_playground import cli

    for name, date in [('first', '260929'), ('second', '260930')]:
        run = experiment_directory(tmp_path, name, date=date)
        run.mkdir(parents=True)
        (run / 'state.json').write_text(json.dumps(dict(status='completed', updated_at=datetime.now(timezone.utc).isoformat())))
    monkeypatch.setattr(cli, 'ROOT', Path(tmp_path))
    cli.main(['status'])
    rows = json.loads(capsys.readouterr().out)
    assert {row['run_id'] for row in rows} == {'first', 'second'}


def test_cli_replay_uses_relocated_original_rollouts(tmp_path, monkeypatch, capsys):
    from drone_playground import cli
    from drone_playground.visualization import rscope_io

    moved = tmp_path / 'experiments/tmp/260930/replay-run/rollouts'
    moved.mkdir(parents=True)
    observed = []
    monkeypatch.setattr(rscope_io, 'publish_run', lambda path: observed.append(path) or path)
    cli.main(['replay', '--directory', str(tmp_path / 'experiments/replay-run/rollouts')])
    assert observed == [moved]


def test_play_resolves_relocated_replay_before_dispatch(tmp_path, monkeypatch):
    from drone_playground import composition

    moved = tmp_path / 'experiments/tmp/260930/replay-run/rollouts'
    moved.mkdir(parents=True)
    original = str(tmp_path / 'experiments/replay-run/rollouts')
    config = dict(mode='play', replay=dict(directory=original))
    monkeypatch.setattr(composition, '_run_experiment', lambda config, root, run_id: config)
    result = composition.run_experiment(config, tmp_path, 'view')
    assert result['replay']['directory'] == str(moved)
    assert config['replay']['directory'] == original
