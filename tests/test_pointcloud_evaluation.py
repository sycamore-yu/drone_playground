"""Frozen recurrent evaluation, collision timing and native replay round trips."""

import importlib
import importlib.util
import json

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import build_environment
from tests.reference_configs import compose_reference as compose_config


def evaluator():
    name = "drone_playground.evaluation.pointcloud"
    assert importlib.util.find_spec(name), "Independent point-cloud evaluation is required"
    return importlib.import_module(name)


def task_and_bank():
    cfg = compose_config(
        "paper_pointcloud",
        [
            "training.device=cpu",
            "training.num_envs=1",
            "scene.obstacles_per_kind=1",
            "observation.sensor.azimuth_count=12",
            "observation.sensor.elevation_count=3",
            "task.duration=0.3",
        ],
    )
    task = build_environment(cfg, "cpu")
    bank = task.scene.sample(jax.random.PRNGKey(1), 1)
    bank = bank.replace(
        active=jnp.array([[True, False, False]]),
        kind=jnp.array([[2, 2, 2]]),
        size=jnp.array([[[0.02, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        origin=jnp.array([[[2.0, 0.0, 3.0], [30.0, 0.0, 3.0], [32.0, 0.0, 3.0]]]),
        start=jnp.array([[0.5, 0.0, 3.0]]),
        goal=jnp.array([[40.0, 0.0, 3.0]]),
        subtype_names=("fixture",),
    )
    return cfg, task, bank


def test_fast_thin_bar_collision_is_detected_and_terminal_state_is_frozen():
    ev = evaluator()
    _, task, bank = task_and_bank()
    state = task.initial_state(bank).replace(vel=jnp.array([[40.0, 0.0, 0.0]]))
    out = ev.advance_checked(
        task, bank, state, jnp.zeros((1, 3)), jnp.zeros(1), jnp.zeros(1, jnp.int32)
    )
    physical, timestamp, outcome, clearance = out
    assert int(outcome[0]) == 2
    assert 0 < float(timestamp[0]) < 0.1
    assert 1.85 < float(physical.pos[0, 0]) < 2.0
    assert float(clearance[0]) < 0
    again = ev.advance_checked(task, bank, physical, jnp.ones((1, 3)), timestamp, outcome)
    np.testing.assert_array_equal(again[0].pos, physical.pos)
    np.testing.assert_array_equal(again[1], timestamp)


def test_numerical_and_out_of_bounds_are_explicit_failures():
    ev = evaluator()
    _, task, bank = task_and_bank()
    state = task.initial_state(bank)
    numerical = ev.advance_checked(
        task, bank, state, jnp.full((1, 3), jnp.nan), jnp.zeros(1), jnp.zeros(1, jnp.int32)
    )
    assert int(numerical[2][0]) == 4
    assert np.isfinite(np.asarray(numerical[0].pos)).all()
    escaped = state.replace(pos=jnp.array([[47.99, 0.0, 3.0]]), vel=jnp.array([[40.0, 0.0, 0.0]]))
    bounds = ev.advance_checked(
        task, bank, escaped, jnp.zeros((1, 3)), jnp.zeros(1), jnp.zeros(1, jnp.int32)
    )
    assert int(bounds[2][0]) == 3


def test_frozen_rollout_and_all_outcome_denominators():
    ev = evaluator()
    from drone_playground.evaluation.tracking import tree_digest
    from drone_playground.learning.algorithms.pointcloud_bptt import initialize

    cfg, task, bank = task_and_bank()
    state, network, _ = initialize(task, cfg)
    before = tree_digest(state.params)
    run = ev.make_rollout(task, network, bank)
    a = jax.tree.map(np.asarray, run(state.params, jnp.array([4.0])))
    b = jax.tree.map(np.asarray, run(state.params, jnp.array([4.0])))
    np.testing.assert_array_equal(a["pos"], b["pos"])
    assert tree_digest(state.params) == before
    report = ev.summarize_trace(a, ["fixture"], 4.0, task.duration, bank.start)
    assert report["num_trials"] == 1
    assert (
        sum(
            report[k]
            for k in ("arrived", "collision", "out_of_bounds", "numerical_failure", "timeout")
        )
        == 1
    )


def test_replay_preserves_point_mass_identity_and_positions(tmp_path):
    ev = evaluator()
    from rscope import rollout

    from drone_playground.learning.algorithms.pointcloud_bptt import initialize

    cfg, task, bank = task_and_bank()
    state, network, _ = initialize(task, cfg)
    task.bank = bank
    trace = jax.tree.map(
        np.asarray, ev.make_rollout(task, network, bank)(state.params, jnp.array([4.0]))
    )
    path = ev.export_case(task, trace, 0, tmp_path)
    assert path.suffix == ".mj_unroll"
    identity = json.loads((tmp_path / "components.json").read_text())
    assert "PointMassLag" in identity["physics_engine"]
    rollout.rollouts.clear()
    rollout.num_evals = 0
    rollout.append_unroll(path)
    loaded = rollout.rollouts[-1]
    length = int(np.asarray(trace["active"])[:, 0].sum())
    np.testing.assert_allclose(loaded.mocap_pos[0, 0, 0], bank.start[0], atol=1e-6)
    np.testing.assert_allclose(loaded.mocap_pos[1:, 0, 0], trace["pos"][:length, 0], atol=1e-6)
    assert loaded.metrics["action/0"].shape[0] == length + 1
    proof = json.loads((tmp_path / "readback-verification.json").read_text())
    assert proof["frames"] == length + 1 and proof["transitions"] == length
    assert proof["positions_and_action_channels_match"]


def test_navigation8_recipe_uses_eight_unique_verified_catalog_cases():
    evaluator()
    cfg = compose_config("paper_pointcloud_navigation8", ["training.device=cpu"])
    assert cfg["mode"] == "eval" and cfg["env"]["task"]["duration"] == 40.0
    task = build_environment(cfg, "cpu", "heldout")
    bank, manifest = task.scene.build()
    assert bank.num_instances == 8
    assert list(manifest["scene_ids"]) == ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
    assert (
        manifest["catalog_sha256"]
        == "18866d707f000dac638e5b5afe8ff6efe45121f8ae4a302daedec7fe677f87a2"
    )
    np.testing.assert_allclose(bank.start[:, 0], 2.0)
    np.testing.assert_allclose(bank.goal[:, 0], 98.0)


def test_collision_free_evaluation_keeps_the_paper_training_transition():
    ev = evaluator()
    _, task, bank = task_and_bank()
    bank = bank.replace(active=jnp.zeros_like(bank.active))
    state = task.initial_state(bank).replace(vel=jnp.array([[1.0, 0.2, 0.0]]))
    command = jnp.array([[2.0, -1.0, 0.3]])
    expected = task.model.step(state, command, task.dt)
    actual = ev.advance_checked(task, bank, state, command, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    np.testing.assert_allclose(actual[0].vector(), expected.vector(), atol=1e-6, rtol=1e-6)


def test_training_run_budget_and_selected_checkpoint_age_are_separate(tmp_path):
    ev = evaluator()
    assert hasattr(ev, "training_provenance"), "Full-run budget needs its own authenticated source"
    path = tmp_path / "selected.pkl"
    path.write_bytes(b"checkpoint")
    result = dict(
        status="completed",
        actual_updates=50000,
        target_updates=50000,
        actual_steps=256000000,
        target_steps=256000000,
        full_budget_completed=True,
        selected=dict(checkpoint=str(path), updates=45000, parameter_sha256="abc"),
    )
    (tmp_path / "result.json").write_text(json.dumps(result))
    provenance = ev.training_provenance(tmp_path, path, 45000, "abc")
    assert provenance["actual_updates"] == 50000
    assert provenance["selected_checkpoint_updates"] == 45000
    assert provenance["full_budget_completed"]


def test_evaluation_summary_keeps_failures_and_training_provenance(tmp_path):
    ev = evaluator()
    assert hasattr(ev, "write_summary"), "A readable result must accompany the machine report"
    report = dict(
        num_trials=1,
        arrived=0,
        collision=1,
        out_of_bounds=0,
        numerical_failure=0,
        timeout=0,
        success_rate=0.0,
        trained_updates=1000,
        training_budget_completed=False,
        training_run_evidence=None,
        checkpoint="/tmp/selected.pkl",
        parameter_sha256="abc",
        scene_ids=["S01"],
        episodes=[
            dict(
                scene_id="S01",
                command_speed_m_s=4.0,
                outcome="collision",
                elapsed_s=2.0,
                arrival_time_s=None,
                path_length_m=3.0,
                final_goal_distance_m=90.0,
                min_clearance_m=-0.01,
            )
        ],
    )
    ev.write_summary(report, tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "碰撞" in text and "1000" in text and "S01" in text
    assert "collision" in (tmp_path / "episodes.csv").read_text()


def test_first_step_numerical_failure_keeps_a_readable_two_state_replay(tmp_path):
    ev = evaluator()
    from rscope import rollout

    _, task, bank = task_and_bank()
    task.bank = bank

    class FailingPolicy:
        hidden_size = 192

        @staticmethod
        def apply(params, points, valid, proprio, hidden):
            return jnp.full((*proprio.shape[:-1], 3), jnp.nan), hidden

    trace = jax.tree.map(
        np.asarray, ev.make_rollout(task, FailingPolicy(), bank)(None, jnp.array([4.0]))
    )
    assert int(trace["active"][:, 0].sum()) == 1
    assert int(trace["outcome"][0, 0]) == 4
    path = ev.export_case(task, trace, 0, tmp_path)
    rollout.rollouts.clear()
    rollout.num_evals = 0
    rollout.append_unroll(path)
    loaded = rollout.rollouts[-1]
    assert loaded.time.shape[0] == 2
    assert float(loaded.time[0, 0]) == 0.0
    np.testing.assert_allclose(loaded.mocap_pos[0, 0, 0], bank.start[0], atol=1e-6)
    assert np.isfinite(loaded.mocap_pos).all()
    proof = json.loads((tmp_path / "readback-verification.json").read_text())
    assert proof["frames"] == 2 and proof["transitions"] == 1
    report = ev.summarize_trace(trace, ["fixture"], 4.0, task.duration, bank.start)
    assert report["num_trials"] == 1 and report["numerical_failure"] == 1
    rollout.rollouts.clear()
    rollout.num_evals = 0
