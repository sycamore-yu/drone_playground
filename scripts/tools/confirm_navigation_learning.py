"""Confirm frozen depth/point-cloud recipes on fixed Navigation8 with actual reset perturbations."""

import argparse
import copy
import subprocess
from pathlib import Path

from confirm_control_learning import ROOT, read, run_job, write


def confirm(manifest_path):
    manifest = read(manifest_path)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if revision != manifest['source_revision']:
        raise ValueError('Navigation confirmation requires its frozen source revision')
    directory = ROOT / 'experiments' / manifest['batch_id']
    frozen = directory / 'manifest.json'
    if frozen.exists() and read(frozen) != manifest:
        raise ValueError('Existing confirmation manifest differs')
    write(frozen,manifest)
    rows = []
    for entry in manifest['entries']:
        for seed in manifest['training_seeds']:
            name = entry['name']
            base = copy.deepcopy(entry['config'])
            base['runtime']['device'] = 'gpu'
            if seed == 0:
                checkpoint = Path(entry['seed0_checkpoint'])
            else:
                base['training']['seed'] = seed
                base.update(mode='train',checkpoint=None)
                training = run_job(directory,f"{manifest['batch_id']}-{name}-seed{seed}-train",base)
                selected = read(training/'result.json')['selected']
                if selected['selection_split'] != 'dev':
                    raise ValueError('Policy selection must use development episodes')
                checkpoint = Path(selected['checkpoint'])
            metadata = read(checkpoint.with_suffix('.json'))
            if metadata['config']['training']['seed'] != seed:
                raise ValueError('Selected policy has the wrong training seed')
            evaluation = copy.deepcopy(metadata['config'])
            evaluation.update(mode='eval',checkpoint=str(checkpoint))
            evaluation['runtime']['device'] = 'gpu'
            evaluation['evaluation'].update(manifest['evaluation'])
            result = run_job(directory,f"{manifest['batch_id']}-{name}-seed{seed}-heldout",evaluation)
            validation = read(result/'eval/release-validation.json')
            validation.update(name=name,training_seed=seed,run_id=result.name,
                              checkpoint_sha256=metadata['sha256'])
            rows.append(validation)
            write(directory/'progress.json',rows)
            print(name,seed,{k:v['success_rate'] for k,v in validation['tasks'].items()},flush=True)
    cells = {entry['name']+'-'+task: all(row['tasks'][task]['passed'] for row in rows
                                      if row['name']==entry['name'])
             for entry in manifest['entries'] for task in ('static','dynamic')}
    write(directory/'result.json',dict(cells=cells,rows=rows,completed=True,passed=all(cells.values()),
                                      rule='Each learning cell must pass for all three training seeds'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    confirm(parser.parse_args().manifest)
