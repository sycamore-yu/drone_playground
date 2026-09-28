"""Frozen-policy evaluation of the original LOTF state tasks on physical time."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.models.gradients import physical_view

from .tracking import summarize_trials, tree_digest


class LOTFEvaluator:
    def __init__(self, task, make_policy, seeds):
        self.env, self.make_policy, self.seeds = task, make_policy, list(seeds)
        if not self.seeds:
            raise ValueError("Evaluation requires at least one trial")
        keys = jax.vmap(jax.random.key)(jnp.asarray(self.seeds))
        self.initial = jax.vmap(task.reset)(keys)
        self._run = jax.jit(self._rollout)

    def _rollout(self, params):
        task = self.env
        policy = self.make_policy(params, deterministic=True)
        initial, observation = self.initial
        batch = len(self.seeds)

        def step(carry, index):
            state, obs, alive = carry
            actions, _ = policy(obs, jax.random.key(0))
            keys = jax.random.split(jax.random.key(0), batch)
            transition = jax.vmap(task.raw_step)(state, actions, keys)

            def finite_moments(value):
                values = value.reshape((batch, -1))
                return jnp.all(jnp.isfinite(values), axis=1) & jnp.isfinite(
                    jnp.sum(values**2, axis=1)
                )

            proposed = transition.state.quadrotor_state
            numerical_valid = finite_moments(actions) & finite_moments(transition.obs)
            numerical_valid &= finite_moments(transition.reward)
            for field in ("p", "R", "v", "omega", "domega", "motor_omega", "acc", "res_acc_mean"):
                numerical_valid &= finite_moments(getattr(proposed, field))
            numerical_valid &= finite_moments(jax.vmap(task.goal)(transition.state))
            invalid = ~numerical_valid
            advance = alive & numerical_valid

            def freeze(proposed, old):
                mask = advance.reshape(advance.shape + (1,) * (proposed.ndim - advance.ndim))
                return jnp.where(mask, proposed, old)

            # An invalid integrator transition ends its trial. Preserve the last
            # finite physical pose for diagnostics/replay; never count it as success.
            nxt = jax.tree.map(freeze, transition.state, state)
            obs_new = jnp.where(advance[:, None], transition.obs, obs)
            view = jax.vmap(physical_view)(nxt.quadrotor_state)
            goal = jax.vmap(task.goal)(nxt)
            error = jnp.linalg.norm(view["pos"] - goal, axis=-1)
            failed = (~alive) | transition.terminated | invalid
            recorded_actions = jnp.where(jnp.isfinite(actions), actions, 0.0)
            metrics = {
                "tracking_error": error,
                "squared_error": error**2,
                "failure": failed.astype(jnp.float32),
                "numerical_failure": (alive & invalid).astype(jnp.float32),
                "reference_x": goal[:, 0],
                "reference_y": goal[:, 1],
                "reference_z": goal[:, 2],
                "physical_thrust": jnp.clip(recorded_actions[:, 0], task.low[0], task.high[0]),
                "speed": jnp.linalg.norm(view["vel"], axis=-1),
                "action_saturation": jnp.mean(
                    ((actions <= task.low) | (actions >= task.high)).astype(jnp.float32), axis=-1
                ),
            }
            record = {
                "pos": view["pos"],
                "quat": view["quat"],
                "obs": obs_new,
                "actions": recorded_actions,
                "time": jnp.full((batch,), (index + 1) * task.dt),
                "reward": jnp.where(advance, transition.reward, 0.0),
                "metrics": metrics,
                "active": alive,
                "failed": failed,
            }
            return (nxt, obs_new, ~failed), record

        return jax.lax.scan(
            step,
            (initial, observation, jnp.ones(batch, dtype=bool)),
            jnp.arange(task.episode_length),
        )[1]

    def run(self, params):
        before = tree_digest(params)
        start = time.monotonic()
        trace = jax.tree.map(np.asarray, self._run(params))
        if before != tree_digest(params):
            raise RuntimeError("Frozen LOTF policy was mutated during evaluation")
        report = summarize_trials(trace, self.seeds, self.env.dt)
        numerical = np.any(trace["metrics"]["numerical_failure"] > 0, axis=0)
        report["numerical_failures"] = int(numerical.sum())
        report["numerical_failure_recording"] = (
            "failed trial with last finite pose; invalid transition reward is unscorable and recorded as zero"
        )
        last_count = min(self.env.freq, self.env.episode_length)
        tail_error = trace["metrics"]["tracking_error"][-last_count:]
        settled = np.sqrt(np.mean(tail_error**2, axis=0))
        valid = np.array([x["completed"] for x in report["episodes"]])
        for i, episode in enumerate(report["episodes"]):
            episode["numerical_failure"] = bool(numerical[i])
            episode["last_second_rmse_m"] = float(settled[i])
            if self.env.task == "lotf_tracking":
                episode["reference_start_index"] = int(self.initial[0].init_ref_traj_idx[i])
        report["last_second_rmse_mean_m"] = float(np.mean(settled))
        if self.env.task == "lotf_hover":
            report["quality_passed"] = bool(
                valid.mean() >= 0.9 and np.any(valid) and settled[valid].mean() <= 0.25
            )
            report["quality_rule"] = (
                "project reporting: >=90% complete, completed last-second position RMSE <=0.25m"
            )
        report.update(
            parameter_sha256=before,
            parameters_frozen=True,
            task=self.env.task,
            dynamics=self.env.dynamics,
            backward=self.env.model.backward,
            drone=self.env.drone,
            elapsed_seconds=time.monotonic() - start,
            episode_length=self.env.episode_length,
            control_delay_seconds=float(self.env.raw.delay),
            components=self.env.component_identity,
            protocol="LOTF native task plus project fixed development/heldout seeds; online adaptation excluded",
        )
        return report, trace
