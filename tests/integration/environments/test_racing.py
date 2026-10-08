"""LSY course events, differentiable task, and correct race denominators."""

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class RacingTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.environments.tasks.racing"
        self.assertIsNotNone(importlib.util.find_spec(name), "Native racing adapter must exist")
        return importlib.import_module(name)

    def env(self):
        env = self.module().RacingEnv(device="cpu", dynamics="so_rpy")
        self.addCleanup(env.close)
        return env

    def test_policy_reference_uses_current_control_tick(self):
        env = self.env()
        state = env.reset(jax.random.PRNGKey(4))
        data = state.pipeline_state.replace(steps=jnp.array([3], dtype=jnp.int32))
        observation = env.observation(data)
        np.testing.assert_allclose(
            observation[13:16] + observation[:3], env.trajectories[0, 3], atol=1e-7
        )

    def test_development_selection_prefers_complete_course_then_gate_progress(self):
        from drone_playground import benchmarks as train

        self.assertTrue(hasattr(train, "checkpoint_eval_score"))
        score = train.checkpoint_eval_score
        short = {"completed": 0, "gates_passed_mean": 0.0, "rmse_all_mean": 0.1}
        advanced = {"completed": 0, "gates_passed_mean": 4.0, "rmse_all_mean": 0.2}
        complete = {"completed": 1, "gates_passed_mean": 1.0, "rmse_all_mean": 0.5}
        self.assertGreater(score("racing", advanced), score("racing", short))
        self.assertGreater(score("racing", complete), score("racing", advanced))
        self.assertEqual(score("figure8", short), (0, -0.1))

    def test_physical_disturbance_seeds_and_actual_short_shac_update(self):
        from drone_playground.configuration import compose_experiment
        from drone_playground.environments.factory import build_environment

        config = compose_experiment(
            "control/shac", "racing", ["runtime.device=cpu", "runtime.action_delay_ms=null"]
        )
        env = build_environment(config, "cpu", "train", 2)
        self.addCleanup(env.close)

        def rollout(key):
            state = env.reset(key)

            def step(state, _):
                state = env.step(state, env.hover_action)
                return state, state.obs

            _, observations = jax.lax.scan(step, state, None, length=8)
            return observations

        simulate = jax.jit(rollout)
        first = simulate(jax.random.PRNGKey(4))
        repeated = simulate(jax.random.PRNGKey(4))
        other = simulate(jax.random.PRNGKey(5))
        np.testing.assert_array_equal(first, repeated)
        self.assertGreater(float(jnp.max(jnp.abs(first - other))), 1e-8)
        from drone_playground.learning.algorithms import shac

        _, _, result = shac.train(
            env,
            dict(
                policy_updates=2,
                num_envs=2,
                horizon_length=4,
                num_evals=3,
                hidden_sizes=[16, 16],
                critic_updates=2,
                learning_rate=0.001,
                critic_learning_rate=0.001,
                normalize_observations=False,
                seed=4,
            ),
        )
        self.assertEqual(result["actual_steps"], 16)
        self.assertGreater(result["actor_parameter_delta_l2"], 0.0)

    def test_native_gate_plane_direction_aperture_and_duplicate_order(self):
        m = self.module()
        passed = m.gate_passed
        origin, quat = jnp.zeros(3), jnp.array([0.0, 0.0, 0.0, 1.0])
        before, after = jnp.array([-1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0])
        self.assertTrue(passed(after, before, origin, quat, False, (0.45, 0.45)))
        self.assertFalse(passed(before, after, origin, quat, False, (0.45, 0.45)))
        self.assertTrue(passed(before, after, origin, quat, True, (0.45, 0.45)))
        self.assertFalse(
            passed(after.at[1].set(0.3), before.at[1].set(0.3), origin, quat, False, (0.45, 0.45))
        )
        from drone_playground.environments.scenes.racing import load_lsy_config

        self.assertEqual(load_lsy_config().env.track.gate_order, [1, 2, 3, 4, 2])

    def test_real_scene_batch_step_and_gradient(self):
        env = self.env()
        state = env.reset(jax.random.PRNGKey(9))
        self.assertEqual(state.obs.shape, (43,))
        self.assertEqual(env.episode_length, 1500)
        self.assertGreater(env.sim.mj_model.ngeom, 20)
        states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(9), 2))
        step = jax.jit(jax.vmap(env.step))
        out = step(states, jnp.tile(env.hover_action, (2, 1)))
        self.assertTrue(np.isfinite(out.obs).all())
        self.assertIn("gates_passed", out.metrics)

        def loss(a):
            s = env.step(state, a)
            return s.pipeline_state.sim_data.states.pos[0, 0, 2]

        gradient = jax.jit(jax.grad(loss))(env.hover_action)
        self.assertTrue(np.isfinite(gradient).all())

    def test_completion_is_gate_events_not_surviving_horizon(self):
        self.module()
        from drone_playground.evaluation.racing import summarize_race

        trace = {
            "active": np.ones((3, 3), bool),
            "reward": np.ones((3, 3)),
            "metrics": {
                "gates_passed": np.array([[0, 0, 0], [2, 2, 0], [5, 3, 0]]),
                "success": np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
                "collision": np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
                "failure": np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
                "tracking_error": np.full((3, 3), 0.1),
            },
        }
        report = summarize_race(trace, [1, 2, 3], 0.02)
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["num_trials"], 3)
        self.assertEqual(report["collisions"], 1)
        self.assertEqual(report["timeouts"], 1)
        self.assertNotIn("quality_passed", report)

    def test_shard_merge_requires_unique_complete_seed_coverage(self):
        from drone_playground.evaluation import racing

        self.assertTrue(hasattr(racing, "merge_race_reports"))
        good = dict(
            case=0,
            seed=30000,
            completed=True,
            failed=False,
            collision=False,
            time_out=False,
            gates_passed=5,
            rmse_m=0.1,
            completion_time_s=17.0,
            steps=850,
            **{"return": 750.0},
        )
        bad = {
            **good,
            "seed": 30001,
            "completed": False,
            "failed": True,
            "collision": True,
            "gates_passed": 2,
            "completion_time_s": None,
            "rmse_m": 0.2,
        }
        reports = [
            {"episodes": [good], "num_trials": 1, "actual_steps": 850},
            {"episodes": [bad], "num_trials": 1, "actual_steps": 850},
        ]
        combined = racing.merge_race_reports(reports, [30000, 30001])
        self.assertEqual(combined["completed"], 1)
        self.assertEqual(combined["num_trials"], 2)
        self.assertEqual(combined["actual_steps"], 1700)
        self.assertEqual(combined["collisions"], 1)
        with self.assertRaises(ValueError):
            racing.merge_race_reports([reports[0], reports[0]], [30000, 30001])

    def test_course_export_embeds_every_referenced_texture(self):
        env = self.env()
        import xml.etree.ElementTree as ET

        import mujoco

        from drone_playground.visualization.rscope_io import _model_bundle

        with tempfile.TemporaryDirectory() as directory:
            xml, assets = _model_bundle(env.sim, Path(directory))
            document = ET.fromstring(xml.read_bytes())
            for asset in document.findall("./asset/texture") + document.findall("./asset/mesh"):
                if "file" in asset.attrib:
                    name = asset.attrib["file"]
                    self.assertFalse(Path(name).is_absolute(), "Portable assets must be relative")
                    self.assertTrue(name in assets, f"Missing portable asset: {name}")
            rebuilt = mujoco.MjModel.from_xml_string(xml.read_text(), assets=assets)
            self.assertEqual(rebuilt.ngeom, env.sim.mj_model.ngeom)
            np.testing.assert_array_equal(rebuilt.tex_data, env.sim.mj_model.tex_data)
            for name in (
                "geom_type",
                "geom_size",
                "geom_pos",
                "geom_quat",
                "geom_rgba",
                "geom_contype",
                "geom_conaffinity",
            ):
                np.testing.assert_allclose(
                    getattr(rebuilt, name), getattr(env.sim.mj_model, name), atol=1e-6, err_msg=name
                )
            # MjSpec.to_xml moves the _dummy body after attached frames; compare
            # named body properties and, critically, the unchanged mocap indices.
            for index in range(env.sim.mj_model.nbody):
                name = mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, index)
                target = mujoco.mj_name2id(rebuilt, mujoco.mjtObj.mjOBJ_BODY, name) if name else 0
                self.assertGreaterEqual(target, 0)
                for prop in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_mocapid"):
                    np.testing.assert_allclose(
                        getattr(rebuilt, prop)[target],
                        getattr(env.sim.mj_model, prop)[index],
                        atol=1e-6,
                        err_msg=f"{name}/{prop}",
                    )
