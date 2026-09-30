"""Organize working results by date and expose the explicitly selected release cells.

Original JSON/checkpoint bytes are preserved. Frozen source trees remain in
place; their runs are linked into the current project's dated working view.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from drone_playground.runs.layout import find_experiment, resolve_artifact  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_date(path):
    for name in ('manifest.json', 'state.json', 'result.json'):
        file = path / name
        if not file.is_file():
            continue
        record = read(file)
        value = record.get('started_at') if isinstance(record, dict) else None
        if value:
            return datetime.fromisoformat(value).astimezone(timezone.utc).strftime('%y%m%d'), 'started_at'
    match = re.search(r'(20\d{6})', path.name)
    if match:
        return datetime.strptime(match[1], '%Y%m%d').strftime('%y%m%d'), 'directory_name'
    return datetime.now(timezone.utc).strftime('%y%m%d'), 'organization_date; original date unavailable'


def process_alive(state):
    identity = state.get('process', {})
    pid = identity.get('pid')
    try:
        ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (OSError, IndexError):
        return False
    return identity.get('start_marker', identity.get('start_ticks')) in (ticks, f'{boot}:{pid}:{ticks}')


def move_working_results(root, apply=False):
    experiments = root / 'experiments'
    rows = []
    for source in sorted(experiments.iterdir()):
        if source.name in ('tmp', 'main_result', 'README.md'):
            continue
        state = source / 'state.json'
        if state.is_file() and read(state).get('status') == 'running' and process_alive(read(state)):
            raise RuntimeError(f'An active run must remain in place: {source}')
        date, basis = run_date(source) if source.is_dir() else (datetime.now(timezone.utc).strftime('%y%m%d'), 'organization_date')
        destination = experiments / 'tmp' / date / source.name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        links = []
        if source.is_dir() and not source.is_symlink():
            links = [(p.relative_to(source), p.resolve()) for p in source.rglob('*') if p.is_symlink()]
        rows.append(dict(source=str(source), destination=str(destination), date_basis=basis,
                         nested_links=[(str(relative), str(target)) for relative, target in links]))
    if not apply:
        return rows
    moves = {Path(row['source']): Path(row['destination']) for row in rows}
    for source, destination in moves.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    for row in rows:
        for relative, target in row['nested_links']:
            target = Path(target)
            for source, destination in moves.items():
                if target.is_relative_to(source):
                    target = destination / target.relative_to(source)
                    break
            link = Path(row['destination']) / relative
            link.unlink()
            link.symlink_to(target, target_is_directory=target.is_dir())
    return rows


def link_directory(destination, source):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f'Different source already selected: {destination}')
        return
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


def index_frozen_runs(root, worktrees):
    rows = []
    if not worktrees.exists():
        return rows
    for tree in sorted(worktrees.iterdir()):
        experiments = tree / 'experiments'
        if not experiments.is_dir():
            continue
        sources = [p for p in experiments.iterdir() if p.is_dir() and p.name not in ('tmp', 'main_result')]
        sources += [p for p in (experiments / 'tmp').glob('*/*') if p.is_dir()]
        for source in sorted(sources):
            existing = find_experiment(root, source.name)
            if existing is not None:
                if existing.resolve() != source.resolve():
                    raise ValueError(f'Distinct runs share an identity: {source.name}')
                continue
            date, basis = run_date(source)
            destination = root / 'experiments/tmp' / date / source.name
            link_directory(destination, source)
            rows.append(dict(path=str(destination.relative_to(root)), source=str(source), date_basis=basis))
    return rows


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(source) != digest(destination):
            raise ValueError(f'Existing selected evidence differs: {destination}')
    else:
        shutil.copy2(source, destination)
    return dict(path=str(destination), sha256=digest(destination))


def build_main_results(root, progress, goal, pointcloud_run=None):
    directory = root / 'experiments/main_result' / goal
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for number, cell in enumerate(progress['cells'], 1):
        name = f"{number:02d}-{cell['method']}-{cell['task']}"
        destination = directory / name
        destination.mkdir(exist_ok=True)
        records = cell.get('training_seeds', [])
        if not records:
            evidence = cell.get('evidence')
            run_id = cell.get('run_id') or (evidence.get('run_id') if isinstance(evidence, dict) else None)
            run_id = run_id or cell.get('development', {}).get('run_id')
            if cell['method'] == 'pointcloud':
                run_id = pointcloud_run
            records = [dict(run_id=run_id)] if run_id else []
        chosen = []
        for record in records:
            run = find_experiment(root, record['run_id'])
            if run is None:
                raise FileNotFoundError(record['run_id'])
            config = read(run / 'resolved-config.json')
            development_only = cell['method'] == 'pointcloud' and config.get('mode') == 'train'
            if development_only:
                result = read(run / 'result.json')
                report_source = run / 'eval' / f"update-{result['selected']['updates']:07d}.json"
            else:
                report_source = run / 'eval/report.json'
            report = read(report_source)
            seed = record.get('training_seed', record.get('seed'))
            if seed is None and config.get('method', {}).get('trainable'):
                seed = config['training']['seed']
            date, _ = run_date(run)
            label = date + ('-solver' if seed is None else f'-seed{seed}')
            if development_only:
                label += '-dev'
            item = destination / label
            if (item / 'selection.json').is_file() and read(item / 'selection.json')['run_id'] != run.name:
                item = destination / f'{label}-{run.name}'
            item.mkdir(exist_ok=True)
            files = []
            for source, filename in [(report_source, 'report.json'), (run / 'resolved-config.json', 'config.json'),
                                     (run / 'manifest.json', 'manifest.json'), (run / 'result.json', 'run-result.json'),
                                     (run / 'eval/release-validation.json', 'original-validation.json')]:
                if source.is_file():
                    files.append(copy_verified(source, item / filename))
            checkpoint = report.get('checkpoint')
            if checkpoint:
                checkpoint = resolve_artifact(checkpoint)
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                metadata = read(checkpoint.with_suffix('.json'))
                if digest(checkpoint) != metadata['sha256'] or metadata['parameter_sha256'] != report['parameter_sha256']:
                    raise ValueError(f'Selected weights do not match evaluation: {checkpoint}')
                files.append(copy_verified(checkpoint, item / 'weights' / checkpoint.name))
                files.append(copy_verified(checkpoint.with_suffix('.json'), item / 'weights' / checkpoint.with_suffix('.json').name))
            link_directory(item / 'original-run', run)
            for kind in ['rollouts', 'traces', 'decision-trace']:
                source = run / kind
                if development_only and kind == 'rollouts':
                    source /= f"update-{result['selected']['updates']:07d}"
                if source.is_dir():
                    link_directory(item / kind, source)
            summary = dict(run_id=run.name, training_seed=seed, date=date,
                           source_run=str(run.relative_to(root)), source_revision=read(run/'manifest.json').get('code',{}).get('commit'),
                           development_only=development_only, parameter_sha256=report.get('parameter_sha256'),
                           selected_checkpoint=checkpoint and str(checkpoint), files=files,
                           quality_passed=cell['passed'], status=cell['status'])
            write(item / 'selection.json', summary)
            write(item / 'qualification.json', dict(task=cell['task'], current_status=cell['status'],
                  current_quality_passed=cell['passed'], criterion=cell.get('criterion'),
                  recorded_evidence=record, original_report_quality_passed=report.get('quality_passed'),
                  development_only=development_only))
            chosen.append(summary)
        write(destination / 'summary.json', dict(cell=cell, selected_runs=chosen))
        (destination / 'README.md').write_text(f"# {cell['method']} × {cell['task']}\n\n状态：{cell['status']}；质量通过：{cell['passed']}。\n\n" +
            ('当前仅有开发集最好结果；尚无三个独立训练种子的正式确认。\n\n' if cell['method']=='pointcloud' and not cell['passed'] else '') +
            ('EGO由用户接受交付，原开发各3/4；不声称正式质量通过。\n\n' if cell['status']=='accepted_by_user' else '') +
            ''.join(f"- [{p.name}]({p.name}/)：报告、配置、选定权重和回放入口。\n" for p in sorted(destination.iterdir()) if p.is_dir()))
        index.append(dict(cell=name, method=cell['method'], task=cell['task'], status=cell['status'],
                          quality_passed=cell['passed'], release_requirement_satisfied=bool(cell['passed'] or cell.get('release_requirement_satisfied')), runs=chosen))
    write(directory / 'progress.json', progress)
    write(directory / 'index.json', dict(goal=goal, generated_utc=datetime.now(timezone.utc).isoformat(), cells=index))
    text = (f"# 第一版18格选定结果\n\n{progress['passed_cells']}格按现行规则质量通过，"
            f"EGO{progress['user_accepted_cells']}格由用户接受交付，共"
            f"{progress['release_requirement_satisfied_cells']}/{progress['required_cells']}格；"
            f"剩余{progress['remaining_quality_cells']}格待正式确认。\n\n")
    text += '| 方法 | 任务 | 状态 | 结果 |\n|---|---|---|---|\n'
    for row in index:
        text += f"| {row['method']} | {row['task']} | {row['status']} | [{row['cell']}]({row['cell']}/README.md) |\n"
    text += '\n原报告原样复制，权重摘要验证；完整回放通过链接访问冻结原始数据。选定结果独立备份继续保留在项目外归档。\n'
    (directory / 'README.md').write_text(text)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--worktrees', type=Path)
    parser.add_argument('--goal', default='v1-18-cells')
    parser.add_argument('--pointcloud-run')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    worktrees = args.worktrees or root.parent / '.worktrees/drone_playground'
    moves = move_working_results(root, apply=args.apply)
    if not args.apply:
        print(json.dumps(dict(planned_moves=len(moves), moves=moves), ensure_ascii=False))
        return
    links = index_frozen_runs(root, worktrees)
    cells = build_main_results(root, read(root/'docs/verification/release-progress.json'), args.goal, args.pointcloud_run)
    migration = root/'experiments/tmp/layout-migration.json'
    previous = read(migration) if migration.exists() else dict(moved=[], linked=[])
    write(migration, dict(moved=[*previous['moved'], *moves], linked=[*previous['linked'], *links],
                                                           created_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(moved=len(moves), indexed_frozen_runs=len(links), cells=len(cells),
                          selected_run_records=sum(len(row['runs']) for row in cells)), ensure_ascii=False))


if __name__ == '__main__':
    main()
