"""Date-grouped working results and read access to relocated legacy artifacts."""

from datetime import datetime, timezone
from pathlib import Path


def _run_id(run_id):
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..', 'tmp', 'main_result'):
        raise ValueError('Run identity must be one directory name')


def iter_experiments(root):
    experiments = Path(root) / 'experiments'
    legacy = [p for p in experiments.glob('*')
              if p.name not in ('tmp', 'main_result') and p.is_dir()]
    dated = [p for p in (experiments / 'tmp').glob('*/*') if p.is_dir()]
    return sorted([*legacy, *dated])


def find_experiment(root, run_id):
    _run_id(run_id)
    experiments = Path(root) / 'experiments'
    candidates = [experiments / run_id, *(experiments / 'tmp').glob('*/' + run_id)]
    existing = {p.resolve(): p for p in candidates if p.is_dir()}
    if len(existing) > 1:
        raise ValueError(f'Ambiguous run identity: {run_id}')
    return next(iter(existing.values()), None)


def experiment_directory(root, run_id, date=None):
    """Return an existing run or the dated location for a new one."""
    existing = find_experiment(root, run_id)
    if existing is not None:
        return existing
    date = date or datetime.now(timezone.utc).strftime('%y%m%d')
    if len(date) != 6 or not date.isdigit():
        raise ValueError('Experiment date must be YYMMDD')
    datetime.strptime(date, '%y%m%d')
    return Path(root) / 'experiments' / 'tmp' / date / run_id


def resolve_artifact(path):
    """Read old experiment paths through the new layout without editing evidence."""
    path = Path(path)
    if path.exists():
        return path
    absolute = path.absolute()
    for parent in absolute.parents:
        if parent.name != 'experiments':
            continue
        relative = absolute.relative_to(parent)
        if not relative.parts or relative.parts[0] in ('tmp', 'main_result'):
            continue
        run = find_experiment(parent.parent, relative.parts[0])
        if run is not None:
            return run.joinpath(*relative.parts[1:])
    return path
