"""First-release control criteria, separate from historical 90% tracking reports."""

import math


def validate_control_report(report, task, minimum_episodes=100):
    if task not in ('tracking', 'racing'):
        raise ValueError('This release validator covers tracking and racing only')
    rows = report['episodes']
    if report.get('split') != 'heldout' or not report.get('parameters_frozen'):
        raise ValueError('Release evidence requires a frozen policy and heldout split')
    native = report.get('parameter_identity_kind') == 'resolved optimization configuration'
    if native and not report.get('runtime_identity'):
        raise ValueError('Optimization release requires its actual solver runtime identity')
    if len(rows) != report['num_trials'] or len(rows) < minimum_episodes:
        raise ValueError('Insufficient or incomplete release episode evidence')
    seeds = [row['seed'] for row in rows]
    if len(set(seeds)) != len(rows):
        raise ValueError('Repeated evaluation seeds are not independent cases')
    if any(bool(row['completed']) == bool(row['failed']) for row in rows):
        raise ValueError('Each episode must record either completion or failure')
    if any(not math.isfinite(row['rmse_m']) or row['rmse_m'] < 0 for row in rows):
        raise ValueError('Every episode must retain finite valid-trajectory RMSE')
    completed = [row for row in rows if row['completed']]
    threshold = .95 if task == 'tracking' else .90
    rate = len(completed) / len(rows)
    rmse = sum(row['rmse_m'] for row in completed) / len(completed) if completed else None
    passed = rate >= threshold and (task != 'tracking' or (rmse is not None and rmse <= .25))
    return dict(protocol='release-control-v1', task=task, passed=passed,
                num_trials=len(rows), completed=len(completed), failed=len(rows)-len(completed),
                completion_rate=rate, minimum_completion_rate=threshold,
                rmse_completed_mean=rmse,
                rmse_all_mean=sum(row['rmse_m'] for row in rows) / len(rows),
                error_rule='Completed mean RMSE <= 0.25m for tracking; all failures retained',
                parameter_sha256=report['parameter_sha256'],
                caveat=('Frozen solver; no learning seeds required' if native else
                        'One frozen policy; the cell additionally requires all 3 training seeds'))
