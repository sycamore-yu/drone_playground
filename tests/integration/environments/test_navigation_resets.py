import numpy as np
import pytest

from drone_playground.configuration import compose_experiment, validate_config
from drone_playground.environments.factory import build_environment
from drone_playground.evaluation.navigation.cases import navigation_resets


@pytest.fixture(scope="module")
def setup():
    config = compose_experiment("navigation/differentiable_pointcloud")
    validate_config(config)
    env = build_environment(config, "cpu", role="eval", count=1)
    yield env.task, config["evaluation"]["initial_conditions"]
    env.close()


def test_frozen_resets_are_safe_reproducible_and_have_actual_variation(setup):
    task, spec = setup
    ids = np.arange(task.bank.num_instances)
    seeds = np.arange(30000, 30000 + len(ids))
    first = navigation_resets(task.bank, ids, seeds, spec, task.body_radius)
    second = navigation_resets(task.bank, ids, seeds, spec, task.body_radius)
    changed = navigation_resets(task.bank, ids, seeds + 1000, spec, task.body_radius)
    np.testing.assert_array_equal(first["position"], second["position"])
    assert not np.array_equal(first["position"], changed["position"])
    assert np.min(first["record"]["clearance_m"]) >= 0.15
    assert np.all(
        np.abs(first["position"] - task.bank.start)
        <= np.asarray(spec["position_half_width_m"]) + 1e-6
    )
    np.testing.assert_allclose(np.linalg.norm(first["quaternion"], axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(first["rotation"]), 1.0, atol=1e-6)
    # Batching/order must not change a case's independent seed interpretation.
    reverse = navigation_resets(task.bank, ids[::-1], seeds[::-1], spec, task.body_radius)
    np.testing.assert_array_equal(first["position"], reverse["position"][::-1])


def test_evaluator_starts_from_frozen_physical_state_and_records_values(setup):
    import copy

    import jax.numpy as jnp

    from drone_playground.evaluation.navigation.recurrent import RecurrentNavigationEvaluator

    class ZeroPolicy:
        hidden_size = 1

        def apply(self, params, points, valid, proprio, memory):
            return jnp.zeros((len(proprio), 3)), memory

    original, spec = setup
    task = copy.copy(original)
    task.duration = task.dt
    evaluator = RecurrentNavigationEvaluator(task, ZeroPolicy(), repeats=1, initial_conditions=spec)
    report, trace = evaluator.run({})
    np.testing.assert_allclose(
        trace["observation_pos"][0], report["initial_conditions"]["position_m"]
    )
    assert report["initial_conditions"]["seeds"] == report["reset_seeds"]
    assert len({tuple(row["initial_position_m"]) for row in report["episodes"]}) == 8
    for row in report["episodes"]:
        assert 13 <= row["delay_ticks"] <= 25
    assert "initial pose/velocity" in report["evaluation_scope"]


def test_depth_method_accepts_same_navigation_protocol():
    config = compose_experiment("papers/depth_diffphysics")
    validate_config(config)


def test_navigation_benchmark_counts_both_tasks_and_keeps_all_failures():
    import copy

    from drone_playground.benchmarks import validate_navigation_report

    ids = ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"] * 25
    rows = [
        dict(
            scene_id=name,
            seed=30000 + i,
            arrived=True,
            outcome="arrived",
            initial_position_m=[i * 0.001, 0, 1],
        )
        for i, name in enumerate(ids)
    ]
    report = dict(
        role="eval",
        parameters_frozen=True,
        parameter_sha256="test",
        num_trials=200,
        episodes=rows,
        initial_conditions=dict(
            seeds=[row["seed"] for row in rows],
            position_m=[row["initial_position_m"] for row in rows],
        ),
    )
    assert validate_navigation_report(report)["passed"]
    failed = copy.deepcopy(report)
    for row in [r for r in failed["episodes"] if r["scene_id"].startswith("D")][:11]:
        row.update(arrived=False, outcome="collision")
    result = validate_navigation_report(failed)
    assert result["tasks"]["static"]["passed"] and not result["tasks"]["dynamic"]["passed"]
    failed["episodes"].pop()
    with pytest.raises(ValueError, match=r"Missing"):
        validate_navigation_report(failed)


@pytest.mark.parametrize("dynamic", [False, True])
def test_native_cases_cover_four_scenes_and_match_learning_reset_seed_meanings(dynamic):
    from collections import Counter

    from drone_playground.environments.scenes.catalog import CatalogNavigationScene
    from drone_playground.evaluation.navigation.cases import navigation_cases

    prefix = "D" if dynamic else "S"
    names = [prefix + suffix for suffix in ("01", "02", "03", "06")]
    bank, _ = CatalogNavigationScene(dynamic=dynamic, scene_ids=names).build(30000, 2)
    groups = navigation_cases(bank, 25, 30000, per_scene=True)
    rows = [case for group in groups.values() for case in group]
    assert len(rows) == 100 and len({r["seed"] for r in rows}) == 100
    assert Counter(row["scene_id"] for row in rows) == dict.fromkeys(names, 25)
    for row in rows:
        assert bank.labels(row["scenario_id"])["subtype"] == row["scene_id"]
        assert (row["seed"] - 30000) % 8 == names.index(row["scene_id"]) + 4 * dynamic


def test_native_benchmark_accepts_exactly_one_complete_task():
    from drone_playground.benchmarks import validate_navigation_report

    rows = [
        dict(
            scene_id="S" + suffix,
            seed=30000 + 8 * r + i,
            arrived=True,
            outcome="arrived",
            initial_position_m=[r * 0.01, i * 0.01, 1.0],
        )
        for r in range(25)
        for i, suffix in enumerate(("01", "02", "03", "06"))
    ]
    report = dict(
        role="eval",
        parameters_frozen=True,
        parameter_sha256="test",
        num_trials=100,
        episodes=rows,
        initial_conditions=dict(
            seeds=[r["seed"] for r in rows], position_m=[r["initial_position_m"] for r in rows]
        ),
    )
    assert validate_navigation_report(report, tasks=("static",))["tasks"]["static"]["passed"]
    with pytest.raises(ValueError, match=r"Missing"):
        validate_navigation_report(report)
