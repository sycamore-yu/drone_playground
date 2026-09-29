"""Run frozen control recipes in an isolated source tree; no hyperparameter search.

Manifest entries contain a complete resolved configuration and a seed-0 candidate.
Each child is a fresh interpreter, so GPU memory and compiler state are released.
All experiment data stay in ROOT/experiments (which may link to the owning repo).
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    temp.replace(path)


def child(job):
    from drone_playground.composition import run_experiment

    value = read(job)
    config = value['config']
    expected = value['config_sha256']
    actual = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if actual != expected:
        raise ValueError('Frozen configuration digest differs')
    run_experiment(config, ROOT, value['run_id'])


def run_job(directory, run_id, config):
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src', 'configs', 'scripts'],
                   cwd=ROOT, check=True)
    path = ROOT / 'experiments' / run_id
    if path.exists():
        if read(path / 'state.json')['status'] == 'completed':
            return path
        raise RuntimeError(f'Existing incomplete run requires inspection: {path}')
    job = directory / (run_id + '.json')
    write(job, dict(run_id=run_id, config=config,
                    config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()))
    env = dict(os.environ, JAX_PLATFORMS='cuda,cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               PYTHONPATH=str(ROOT / 'src'))
    with job.with_suffix('.log').open('w') as output:
        result = subprocess.run([sys.executable, __file__, '--job', str(job)], cwd=ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT, timeout=3900)
    if result.returncode:
        raise RuntimeError(f'Confirmation job failed: {run_id}; inspect {job.with_suffix(".log")}')
    return path


def confirm(manifest_path):
    from drone_playground.evaluation.release_control import validate_control_report

    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    if revision != manifest['source_revision']:
        raise ValueError('Run confirmation from the declared isolated source revision')
    directory = ROOT / 'experiments' / manifest['batch_id']
    directory.mkdir(parents=True, exist_ok=True)
    # Fail rather than silently reusing a batch with different conditions.
    frozen = directory / 'manifest.json'
    if frozen.exists() and read(frozen) != manifest:
        raise ValueError('Existing confirmation manifest differs')
    write(frozen, manifest)
    rows = []
    for entry in manifest['entries']:
        name = entry['name']
        for seed in manifest['training_seeds']:
            base = copy.deepcopy(entry['config'])
            base['runtime']['device'] = 'gpu'
            if seed == 0:
                checkpoint = ROOT / entry['seed0_checkpoint']
            else:
                base['training']['seed'] = seed
                base['mode'] = 'train'
                base['checkpoint'] = None
                training = run_job(directory, f"{manifest['batch_id']}-{name}-seed{seed}-train", base)
                selection = read(training / 'checkpoints/best.json')
                if selection['selection_split'] != 'dev':
                    raise ValueError('Policy selection must use development episodes')
                checkpoint = training / 'checkpoints' / selection['path']
            metadata = read(checkpoint.with_suffix('.json'))
            # Checkpoint digest is independently checked by load_policy at evaluation.
            if metadata['config']['training']['seed'] != seed:
                raise ValueError('Selected policy has the wrong training seed')
            evaluation = copy.deepcopy(metadata['config'])
            evaluation.update(mode='eval', checkpoint=str(checkpoint))
            evaluation['runtime']['device'] = 'gpu'
            evaluation['evaluation'].update(environment='checkpoint', split='heldout',
                                            episodes=manifest['episodes'], seed_start=manifest['seed_start'])
            result = run_job(directory, f"{manifest['batch_id']}-{name}-seed{seed}-heldout", evaluation)
            validation = validate_control_report(read(result / 'eval/report.json'), entry['task'],
                                                 manifest['episodes'])
            validation.update(name=name, training_seed=seed, run_id=result.name,
                              checkpoint_sha256=metadata['sha256'])
            write(result / 'eval/release-validation.json', validation)
            rows.append(validation)
            write(directory / 'progress.json', rows)
            print(name, seed, validation['passed'], validation['completion_rate'], flush=True)
    cells = {entry['name']: all(row['passed'] for row in rows if row['name'] == entry['name'])
             for entry in manifest['entries']}
    write(directory / 'result.json', dict(cells=cells, rows=rows, completed=True,
          passed=all(cells.values()), rule='All three independently trained policies must pass'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--manifest')
    group.add_argument('--job')
    args = parser.parse_args()
    if args.job:
        child(args.job)
    else:
        confirm(args.manifest)
