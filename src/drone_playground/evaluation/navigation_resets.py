"""Freeze actual initial conditions for fixed-geometry navigation evaluation.

Sampling is on the host with per-case PCG64 streams. Collision checks use the
same scene bank as sensing and dynamics. No seed is treated as new geometry.
"""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.environments.scenes.navigation import clearance_and_collision


def navigation_resets(bank, scene_indices, seeds, specification, body_radius):
    spec = dict(specification)
    if spec['name'] != 'navigation8-initial-v1':
        raise ValueError('Unknown navigation initial-condition distribution')
    position = np.asarray(spec['position_half_width_m'], dtype=float)
    velocity = np.asarray(spec['velocity_half_width_mps'], dtype=float)
    yaw = float(spec['yaw_half_width_rad'])
    clearance = float(spec['minimum_clearance_m'])
    attempts = int(spec['position_candidates'])
    if (position.shape != (3,) or velocity.shape != (3,) or attempts < 1
            or not np.isfinite(np.r_[position, velocity, yaw, clearance, body_radius]).all()
            or np.any(position < 0) or np.any(velocity < 0) or min(yaw,clearance,body_radius) < 0):
        raise ValueError('Invalid navigation initial-condition bounds')
    indices = np.asarray(scene_indices, dtype=np.int32)
    if len(indices) != len(seeds) or len(seeds) == 0:
        raise ValueError('Each initial state requires a scene index and seed')
    proposals, velocities, yaws = [], [], []
    starts = np.asarray(bank.start)[indices]
    for origin, seed in zip(starts, seeds, strict=True):
        pk, vk, yk = [np.random.Generator(np.random.PCG64(key))
                      for key in np.random.SeedSequence(int(seed)).spawn(3)]
        proposals.append(origin + pk.uniform(-1,1,(attempts,3)) * position)
        velocities.append(vk.uniform(-1,1,3) * velocity)
        yaws.append(yk.uniform(-yaw,yaw))
    proposals = np.asarray(proposals, dtype=np.float32)
    # A yaw-only reset preserves the local collision-centre z offset.
    @jax.jit
    def distances(points):
        return jax.vmap(lambda index, row: jax.vmap(
            lambda p: clearance_and_collision(bank,index,0.,p+jnp.array([0.,0.,.005]),body_radius)[0]
        )(row))(jnp.asarray(indices), points)
    distances = np.asarray(distances(jnp.asarray(proposals)))
    inside = np.all((proposals >= np.asarray(bank.world_low)+body_radius)
                    & (proposals <= np.asarray(bank.world_high)-body_radius),axis=-1)
    safe = inside & (distances >= clearance)
    if not safe.any(axis=1).all():
        raise ValueError('No valid initial-state proposal; revise the frozen distribution before evaluation')
    choices = safe.argmax(axis=1)
    selected = proposals[np.arange(len(indices)), choices]
    angles = np.asarray(yaws, dtype=np.float32)
    quaternions = np.stack([np.zeros_like(angles),np.zeros_like(angles),
                            np.sin(angles/2),np.cos(angles/2)],axis=-1)
    rotations = np.zeros((len(indices),3,3),dtype=np.float32)
    rotations[:,0,0] = rotations[:,1,1] = np.cos(angles)
    rotations[:,0,1],rotations[:,1,0],rotations[:,2,2] = -np.sin(angles),np.sin(angles),1.
    return dict(position=selected, velocity=np.asarray(velocities,dtype=np.float32),
                quaternion=quaternions, rotation=rotations,
                record=dict(specification=spec, body_radius_m=body_radius,
                            sampler='NumPy PCG64 / SeedSequence(seed).spawn(3)',
                            seeds=list(map(int,seeds)), scene_indices=indices.tolist(),
                            position_m=selected.tolist(), velocity_mps=np.asarray(velocities).tolist(),
                            quaternion_xyzw=quaternions.tolist(),
                            clearance_m=distances[np.arange(len(indices)),choices].tolist(),
                            rejected_position_proposals=choices.tolist()))


def validate_navigation_report(report, minimum_per_task=100, tasks=("static", "dynamic")):
    """Validate the two Navigation8 cells without hiding failures or missing scenes."""
    if (report.get('split') != 'heldout' or not report.get('parameters_frozen')
            or not report.get('initial_conditions')):
        raise ValueError('Release navigation needs frozen parameters and independent heldout initial conditions')
    rows = report['episodes']
    prefixes = dict(static='S', dynamic='D')
    if not tasks or not set(tasks) <= set(prefixes):
        raise ValueError('Select static and/or dynamic navigation tasks')
    expected = {prefixes[task]+suffix for task in tasks for suffix in ('01','02','03','06')}
    if (len(rows) != report['num_trials'] or set(row['scene_id'] for row in rows) != expected
            or len({row['seed'] for row in rows}) != len(rows)):
        raise ValueError('Missing navigation scenes/episodes or repeated evaluation seeds')
    # Confirm that the recorded actual resets are aligned with the episode rows.
    reset = report['initial_conditions']
    if reset['seeds'] != [row['seed'] for row in rows] or len(reset['position_m']) != len(rows):
        raise ValueError('Initial-condition manifest and episodes differ')
    actual_positions = np.asarray([row['initial_position_m'] for row in rows])
    if (not np.isfinite(actual_positions).all()
            or not np.allclose(actual_positions, reset['position_m'], rtol=0, atol=1e-6)
            or len(np.unique(actual_positions, axis=0)) != len(rows)):
        raise ValueError('Recorded initial positions differ from the manifest or repeat')
    results = {}
    for task in tasks:
        prefix = prefixes[task]
        cases = [row for row in rows if row['scene_id'].startswith(prefix)]
        if len(cases) < minimum_per_task:
            raise ValueError('Each navigation task requires at least 100 heldout episodes')
        for row in cases:
            if row['outcome'] not in ('arrived','collision','out_of_bounds','numerical_failure','timeout'):
                raise ValueError('Unknown navigation outcome')
            if bool(row['arrived']) != (row['outcome'] == 'arrived'):
                raise ValueError('Inconsistent navigation outcome')
        successes = sum(row['arrived'] for row in cases)
        rates = {scene: sum(row['arrived'] for row in cases if row['scene_id']==scene)
                 / sum(row['scene_id']==scene for row in cases) for scene in sorted(expected) if scene.startswith(prefix)}
        results[task] = dict(num_trials=len(cases),arrived=successes,success_rate=successes/len(cases),
                           scene_success_rates=rates,passed=successes/len(cases)>=.9)
    return dict(protocol='release-navigation-v1',tasks=results,passed=all(x['passed'] for x in results.values()),
                geometry_scope='Fixed Navigation8 only; no unseen-geometry claim',
                parameter_sha256=report['parameter_sha256'],
                caveat='Each learning cell additionally requires three training seeds; native solvers require a frozen configuration and runtime identity')


def native_navigation_cases(bank, repeats, seed_start, per_scene=False):
    """Map each native episode explicitly to geometry and an independent reset seed.

    Per-scene seeds match the interleaved eight-scene learning protocol, even
    when static and dynamic native processes are evaluated in separate runs.
    """
    from drone_playground.environments.scenes.navigation import DIFFICULTIES

    groups = {difficulty: [] for difficulty in DIFFICULTIES}
    if per_scene:
        canonical = ['S01','S02','S03','S06','D01','D02','D03','D06']
        unique = {}
        for index in range(bank.num_instances):
            label = bank.labels(index)
            unique.setdefault(label['subtype'], (index, label['difficulty']))
        if set(unique) not in (set(canonical[:4]), set(canonical[4:])):
            raise ValueError('Native release requires all four scenes of one navigation task')
        for name in canonical:
            if name not in unique:
                continue
            scenario, difficulty = unique[name]
            for repeat in range(repeats):
                groups[difficulty].append(dict(scenario_id=scenario,
                    seed=seed_start+8*repeat+canonical.index(name), scene_id=name))
    else:
        for di, difficulty in enumerate(DIFFICULTIES):
            for case in range(repeats):
                index = di*repeats+case
                groups[difficulty].append(dict(scenario_id=index, seed=seed_start+case,
                                               scene_id=bank.labels(index)['subtype']))
    return groups
