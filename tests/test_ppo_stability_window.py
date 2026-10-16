"""The stability gate requires both per-scene quality and original-case retention."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stability_window", Path(__file__).resolve().parents[1] / "tools/ppo_stability_window.py"
)
WINDOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WINDOW)


def cases():
    """Six scenes with the exact independent physical initialization identities."""
    return {
        (scene, "geometry", "1500000", str(world)): "SUCCESS"
        for scene in WINDOW.SCENES
        for world in range(25)
    }


def test_budget_rounding_covers_original_collapse_scale():
    """The actual rollout size never rounds the agreed budgets down."""
    assert WINDOW.window_steps(64 * 512) == (28, 4)
    with pytest.raises(ValueError, match="positive integer"):
        WINDOW.window_steps(0)


def test_short_diagnostic_keeps_cadence_and_cannot_exceed_full_budget():
    """A bounded pilot changes its length without relaxing any evaluation threshold."""
    assert WINDOW.window_steps(64 * 512, 8) == (8, 4)
    for updates in (0, -1, 29, 8.5, True):
        with pytest.raises(ValueError, match="within the full window"):
            WINDOW.window_steps(64 * 512, updates)


def test_per_scene_pass_is_insufficient_when_retention_fails():
    """Eight lost cases fail; seven lost cases meet the95percent retention gate."""
    original = cases()
    current = dict(original)
    for scene in WINDOW.SCENES[:4]:
        for world in (0, 1):
            current[(scene, "geometry", "1500000", str(world))] = "TIMEOUT"
    result = WINDOW.quality(current, original)
    assert min(result["per_scene_successes"].values()) == 23
    assert result["retained_initial_successes"] == 142
    assert not result["passed"]
    current[("D01", "geometry", "1500000", "1")] = "SUCCESS"
    assert WINDOW.quality(current, original)["passed"]


def test_retention_pass_is_insufficient_when_one_scene_fails():
    """A weak scene cannot hide behind high aggregate retention."""
    original = cases()
    current = dict(original)
    for world in range(3):
        current[("S03", "geometry", "1500000", str(world))] = "COLLISION"
    result = WINDOW.quality(current, original)
    assert result["retention_fraction"] == 0.98
    assert result["per_scene_successes"]["S03"] == 22
    assert not result["passed"]


def test_changed_case_identity_cannot_masquerade_as_retention():
    """An equally successful different cohort cannot demonstrate preservation."""
    original = cases()
    current = dict(original)
    event = current.pop(("S01", "geometry", "1500000", "0"))
    current[("S01", "geometry", "2000000", "0")] = event
    with pytest.raises(ValueError, match="exact original"):
        WINDOW.quality(current, original)


def test_lr_override_preserves_source_and_all_other_settings():
    """The continuation has a truthful config and leaves its source immutable."""
    source = {
        "experiment": {"learning": {"options": {"lr": 0.0002, "critic_lr": 0.0005}}},
        "learning": {"lr": 0.0002, "critic_lr": 0.0005, "horizon": 512},
    }
    actual = WINDOW.continuation_config(source, 0.1)
    assert actual["learning"]["lr"] == pytest.approx(0.00002)
    assert actual["experiment"]["learning"]["options"]["lr"] == pytest.approx(0.00002)
    actual["learning"]["lr"] = 0.0002
    actual["experiment"]["learning"]["options"]["lr"] = 0.0002
    assert actual == source
    assert WINDOW.continuation_config(source, 1.0) == source


def test_world_expansion_preserves_scene_proportions_and_optimizer_config():
    """More sampled trajectories do not silently change epochs, rates or geometry."""
    source = {
        "experiment": {"simulation": {"num_envs": 64}},
        "learning": {"num_envs": 64, "minibatches": 4, "ppo_epochs": 4},
        "training_sampling": {
            "scene_sampling": "parallel",
            "worlds_per_scene": 8,
            "scenes": list(range(8)),
            "geometry_sha256": {"scene": "unchanged"},
        },
    }
    expanded = WINDOW.expanded_config(source, 4)
    assert expanded["learning"]["num_envs"] == 256
    assert expanded["training_sampling"]["worlds_per_scene"] == 32
    assert source["learning"]["num_envs"] == 64
    for mapping, key in (
        (expanded["learning"], "num_envs"),
        (expanded["experiment"]["simulation"], "num_envs"),
        (expanded["training_sampling"], "worlds_per_scene"),
    ):
        mapping[key] //= 4
    assert expanded == source
    assert WINDOW.window_steps(256 * 512, 2) == (2, 1)
    for multiplier in (0, 1, True, 1.5):
        with pytest.raises(ValueError, match="integer"):
            WINDOW.expanded_config(source, multiplier)
    source["training_sampling"]["scene_sampling"] = "sequential"
    with pytest.raises(ValueError, match="parallel"):
        WINDOW.expanded_config(source, 4)


def test_expanded_inflight_state_executes_real_ppo_without_resetting_learner():
    """Clone physical/history leaves, retain Adam, then sample independent new actions."""
    import jax
    import numpy as np
    from crazyflow.utils import world_mask

    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import make_environment

    environment_options = {
        "task": "navigation",
        "scene": ["S01", "D01"],
        "sensor": "depth",
        "sensor_config": {"width": 64, "height": 48},
        "point_count": 16,
    }
    before_env = make_environment(num_envs=4, **environment_options)
    after_env = make_environment(num_envs=8, **environment_options)
    options = {"horizon": 2, "ppo_epochs": 1, "minibatches": 1}
    try:
        before = Trainer(before_env, kind="depth", config=options)
        after = Trainer(after_env, kind="depth", config=options)
        original, _ = before.update(before.initialize())
        expanded = WINDOW.expand_worlds(original, 4, 2)
        assert expanded.params is original.params
        assert expanded.critic_params is original.critic_params
        assert expanded.optimizer_state is original.optimizer_state
        assert expanded.rng is original.rng
        assert expanded.interactions == original.interactions == 8
        assert expanded.env_state.physics.core.n_worlds == 8
        for old, new, batched in zip(
            jax.tree.leaves(original.env_state.physics),
            jax.tree.leaves(expanded.env_state.physics),
            jax.tree.leaves(world_mask(original.env_state.physics)),
            strict=True,
        ):
            if not batched:
                assert new is old
        for name in ("env_state", "recurrent_memory", "objective_state"):
            old_state, new_state = getattr(original, name), getattr(expanded, name)
            if name == "env_state":
                old_state, new_state = (
                    old_state.replace(physics=None),
                    new_state.replace(physics=None),
                )
            for old, new in zip(
                jax.tree.leaves(old_state),
                jax.tree.leaves(new_state),
                strict=True,
            ):
                expected = np.repeat(old, 2, axis=0) if old.ndim and old.shape[0] == 4 else old
                if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
                    np.testing.assert_array_equal(
                        jax.random.key_data(new), jax.random.key_data(expected)
                    )
                else:
                    np.testing.assert_array_equal(new, expected)
        result, metrics = after.update(expanded)
        assert result.interactions == 24 and result.updates == 2
        assert result.env_state.done.shape == (8,)
        assert all(np.isfinite(value).all() for value in metrics.values())
        positions = np.asarray(result.env_state.physics.states.pos)
        assert not np.array_equal(positions[0], positions[1])
        for optimizer in ("actor", "critic"):
            counts = [
                int(v)
                for v in jax.tree.leaves(result.optimizer_state[optimizer])
                if v.shape == () and np.issubdtype(v.dtype, np.integer)
            ]
            assert counts == [2]
    finally:
        before_env.sim.close()
        after_env.sim.close()


@pytest.mark.parametrize("scale", [0, -1, 2, float("nan"), float("inf")])
def test_lr_override_rejects_freezing_or_nonfinite_rates(scale):
    """The diagnostic must execute finite, nonzero Actor updates."""
    with pytest.raises(ValueError, match="finite"):
        WINDOW.continuation_config({}, scale)


def test_entropy_removal_preserves_source_and_other_learning_settings():
    """No noise freeze or optimizer change accompanies entropy-bonus removal."""
    options = {"lr": 0.0002, "entropy_weight": 0.005, "log_std_init": -2.5}
    source = {"learning": dict(options), "experiment": {"learning": {"options": dict(options)}}}
    actual = WINDOW.continuation_config(source, 1.0, disable_entropy=True)
    assert actual["learning"]["entropy_weight"] == 0
    assert actual["experiment"]["learning"]["options"]["entropy_weight"] == 0
    assert source["learning"]["entropy_weight"] == 0.005
    assert source["experiment"]["learning"]["options"]["entropy_weight"] == 0.005
    actual["learning"]["entropy_weight"] = 0.005
    actual["experiment"]["learning"]["options"]["entropy_weight"] = 0.005
    assert actual == source
    with pytest.raises(ValueError, match="one learning setting"):
        WINDOW.continuation_config(source, 0.1, disable_entropy=True)
    for weight in (0, -1, float("nan"), 0.001):
        source["learning"]["entropy_weight"] = weight
        with pytest.raises(ValueError, match="consistent positive"):
            WINDOW.continuation_config(source, 1.0, disable_entropy=True)


def test_cached_baseline_requires_same_complete_control():
    """Reuse is conditional on checkpoint identity, not an attractive old score."""
    provenance = {
        "config_sha256": "config",
        "source_sha256": "source",
        "updates": 144,
        "interactions": 4718592,
    }
    control = {
        "status": "completed",
        "source_checkpoint": "/run/model_000144",
        "source_config_sha256_reused": "config",
        "source_sha256_reused": "source",
        "origin_update": 144,
        "origin_interactions": 4718592,
        "baseline_cases": [[*key, event] for key, event in cases().items()],
    }
    path = Path(control["source_checkpoint"])
    assert WINDOW.reuse_baseline(control, path, provenance) == cases()
    for changed in (
        {"status": "running"},
        {"actor_lr_scale": 0.1},
        {"diagnostic": True},
        {"source_checkpoint": "/other"},
        {"source_config_sha256_reused": "other"},
    ):
        with pytest.raises(ValueError, match="matched control"):
            WINDOW.reuse_baseline({**control, **changed}, path, provenance)
    control["baseline_cases"].append(control["baseline_cases"][0])
    with pytest.raises(ValueError, match="valid first-pass"):
        WINDOW.reuse_baseline(control, path, provenance)


def test_full_batch_override_changes_only_minibatches():
    """Larger optimizer batches must not silently change LR, entropy or rollout length."""
    options = {"lr": 0.0002, "minibatches": 4, "ppo_epochs": 4, "horizon": 512}
    source = {"learning": dict(options), "experiment": {"learning": {"options": dict(options)}}}
    actual = WINDOW.continuation_config(source, 1.0, full_batch=True)
    assert actual["learning"]["minibatches"] == 1
    assert actual["experiment"]["learning"]["options"]["minibatches"] == 1
    actual["learning"]["minibatches"] = 4
    actual["experiment"]["learning"]["options"]["minibatches"] = 4
    assert actual == source
    for scale, entropy in ((0.1, False), (1.0, True)):
        with pytest.raises(ValueError, match="one learning setting"):
            WINDOW.continuation_config(source, scale, entropy, full_batch=True)
    for batches in (0, 1, 2.5, True, 2):
        source["learning"]["minibatches"] = batches
        with pytest.raises(ValueError, match="consistent multiple"):
            WINDOW.continuation_config(source, 1.0, full_batch=True)


def test_full_batch_changes_actual_adam_count_without_reducing_samples():
    """The native option uses fewer real optimizer steps on the same rollout budget."""
    import jax
    import numpy as np
    import optax

    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import make_environment

    env = make_environment(num_envs=4)
    try:
        control = Trainer(env, config={"horizon": 2, "ppo_epochs": 2, "minibatches": 4})
        full = Trainer(env, config={"horizon": 2, "ppo_epochs": 2, "minibatches": 1})
        initial = control.initialize()
        for trainer, expected_count in ((control, 8), (full, 2)):
            result, metrics = trainer.update(initial)
            assert result.interactions == 8 and result.updates == 1
            for optimizer in ("actor", "critic"):
                counts = [
                    int(leaf)
                    for leaf in jax.tree.leaves(result.optimizer_state[optimizer])
                    if leaf.shape == () and np.issubdtype(leaf.dtype, np.integer)
                ]
                assert counts == [expected_count]
            assert all(np.isfinite(value).all() for value in metrics.values())
            assert (
                float(
                    optax.tree.norm(jax.tree.map(lambda a, b: a - b, result.params, initial.params))
                )
                > 0
            )
    finally:
        env.sim.close()


def test_lr_change_retains_adam_moments_and_scales_actual_actor_step():
    """The real Trainer can continue Adam history with a smaller nonzero step."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import make_environment

    env = make_environment(num_envs=1)
    control = Trainer(env, config={"lr": 0.0002})
    reduced = Trainer(env, config={"lr": 0.00002})
    initial = control.initialize()
    variables = {"params": initial.params, "log_std": initial.log_std}
    gradients = jax.tree.map(jnp.ones_like, variables)
    _, moments = control.actor_optimizer.update(
        gradients, initial.optimizer_state["actor"], variables
    )
    original_step, expected_moments = control.actor_optimizer.update(gradients, moments, variables)
    reduced_step, actual_moments = reduced.actor_optimizer.update(gradients, moments, variables)
    assert float(optax.tree.norm(reduced_step)) > 0
    for expected, actual in zip(
        jax.tree.leaves(original_step), jax.tree.leaves(reduced_step), strict=True
    ):
        np.testing.assert_allclose(actual, expected * 0.1, rtol=1e-6, atol=1e-9)
    for expected, actual in zip(
        jax.tree.leaves(expected_moments), jax.tree.leaves(actual_moments), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
