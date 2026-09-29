"""Certify a frozen solver configuration on independent control-task episodes."""

import argparse
import copy
import hashlib
import subprocess
from pathlib import Path

from confirm_control_learning import ROOT, read, run_job, write


def confirm(manifest_path, method):
    manifest = read(manifest_path)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if revision != manifest['source_revision']:
        raise ValueError('Solver confirmation requires the declared frozen source revision')
    directory = ROOT / 'experiments' / manifest['batch_id']
    frozen = directory / 'manifest.json'
    if frozen.exists() and read(frozen) != manifest:
        raise ValueError('Existing solver confirmation manifest differs')
    write(frozen, manifest)
    rows = []
    for entry in manifest['entries']:
        if entry['method'] != method:
            continue
        development = Path(entry['development_report'])
        if hashlib.sha256(development.read_bytes()).hexdigest() != entry['development_sha256']:
            raise ValueError('Development evidence changed after the solver recipe was frozen')
        report = read(development)
        config = copy.deepcopy(entry['config'])
        for field in ('env', 'method', 'runtime', 'objective'):
            if config[field] != report['config'][field]:
                raise ValueError('Solver confirmation differs from its declared development conditions')
        if (report['num_trials'] < 32 or report['config']['evaluation']['split'] != 'dev'
                or report['completion_rate'] < (.95 if entry['task'] == 'tracking' else .9)
                or (entry['task'] == 'tracking' and report['rmse_completed_mean'] > .25)):
            raise ValueError('Complete the development quality check before certifying this solver')
        config['mode'], config['checkpoint'] = 'eval', None
        config['evaluation'].update(split='heldout', episodes=100, seed_start=30000,
                                    release_validation='control-v1')
        run_id = f"{manifest['batch_id']}-{method}-{entry['task']}"
        run = run_job(directory, run_id, config)
        evidence = read(run / 'eval/release-validation.json')
        rows.append(dict(name=entry['name'], run_id=run_id, **evidence))
        write(directory / f'progress-{method}.json', rows)
        print(entry['name'], evidence['passed'], evidence['completion_rate'], flush=True)
    if not rows:
        raise ValueError('Manifest has no entries for the requested solver')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--method', choices=['attitude_mpc', 'sampling_mpc'], required=True)
    args = parser.parse_args()
    confirm(args.manifest, args.method)
