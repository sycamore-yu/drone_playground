"""Episode-return measurement and optional within-run checkpoint selection."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning.objective import EpisodeObjective
from drone_playground.simulation.factory import create_environment
from drone_playground.simulation.tasks import Event


class RewardSelection:
    """Compare completed episode returns without changing the training stream."""

    def __init__(self, config, trainer, saved=None):
        """Bind the selection recipe and restore its exact measurement history."""
        self.config = config.learning.best
        if self.config.source not in (None, "training", "evaluation"):
            raise ValueError("learning.best.source must be null, training, or evaluation")
        for name in ("interval", "episodes"):
            if type(self.config[name]) is not int or self.config[name] < 1:
                raise ValueError(f"learning.best.{name} must be a positive integer")
        if type(self.config.seed) is not int or not 0 <= self.config.seed < 2**32:
            raise ValueError("learning.best.seed must be an unsigned 32-bit integer")
        self.state = saved or {
            "best_mean_reward": None,
            "best_checkpoint": None,
            "last_interactions": 0,
            "episodes_completed": 0,
            "episode_return_sum": 0.0,
            "episode_length_sum": 0.0,
            "episode_successes": 0,
            "window_episodes": 0,
            "window_return": 0.0,
        }
        self.evaluate = (
            fixed_evaluator(config, trainer) if self.config.source == "evaluation" else None
        )

    def observe(self, metrics):
        """Accumulate actual completed live episodes, including update-spanning ones."""
        count = int(metrics["episodes_completed"])
        self.state["episodes_completed"] += count
        self.state["episode_return_sum"] += metrics["episode_return_sum"]
        self.state["episode_length_sum"] += metrics["episode_length_sum"]
        self.state["episode_successes"] += int(metrics["episode_successes"])
        self.state["window_episodes"] += count
        self.state["window_return"] += metrics["episode_return_sum"]

    def measure(self, state, *, final=False):
        """Evaluate after a completed update; intervals count environment transitions."""
        if self.config.source is None or int(state.interactions) == self.state["last_interactions"]:
            return None
        if not final and (
            int(state.interactions) // self.config.interval
            <= self.state["last_interactions"] // self.config.interval
        ):
            return None
        if self.evaluate is not None:
            rows = self.evaluate(state.params)
            mean = float(np.mean([row["return"] for row in rows]))
            count = len(rows)
        else:
            count = self.state["window_episodes"]
            if not count:
                return None
            rows = []
            mean = self.state["window_return"] / count
        if not math.isfinite(mean):
            raise FloatingPointError("Checkpoint selection produced a nonfinite episode return")
        self.state.update(
            last_interactions=int(state.interactions), window_episodes=0, window_return=0.0
        )
        previous = self.state["best_mean_reward"]
        return {
            "source": self.config.source,
            "mean_reward": mean,
            "episodes": count,
            "episode_records": rows,
            "improved": previous is None or mean > previous,
        }


def fixed_evaluator(config, trainer):
    """Build a deterministic, separate Simulation batch with fixed seeds and difficulty."""
    settings = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    settings.simulation.num_envs = config.learning.best.episodes
    if "procedural" in settings.simulation:
        settings.simulation.procedural.training_seed = config.learning.best.seed
    env = create_environment(settings)
    objective = EpisodeObjective(
        env, trainer.config, recipe=trainer.objective.recipe, loss_name=trainer.loss_name
    )
    initial = env.reset(
        jax.random.PRNGKey(config.learning.best.seed),
        **trainer._reset_options,
        **({"sampling_progress": 1.0} if trainer.distance_curriculum else {}),
    )
    limit = math.ceil(env.task.duration / env.dt) + 1

    @jax.jit
    def run(params):
        carry = (
            initial,
            trainer.actor.initialize_memory(env.num_envs),
            objective.initialize_state(env.num_envs),
            jnp.zeros(env.num_envs),
            jnp.zeros(env.num_envs, jnp.int32),
            jnp.asarray(0),
        )

        def step(carry):
            before, memory, history, returns, lengths, steps = carry
            raw, memory, _ = trainer.actor.apply(params, env.observe(before), memory)
            action = trainer.method.deterministic(raw)
            after = trainer.method.execute(env, before, action)
            cost, history, _ = objective.cost(before, after, action, history)
            active = ~before.done
            return (
                after,
                memory,
                history,
                returns - jnp.where(active, cost, 0),
                lengths + active,
                steps + 1,
            )

        return jax.lax.while_loop(lambda c: (~jnp.all(c[0].done)) & (c[-1] < limit), step, carry)

    def evaluate(params):
        after, _, _, returns, lengths, _ = jax.device_get(run(params))
        if not np.all(after.done) or not np.isfinite(returns).all():
            raise RuntimeError("Fixed reward evaluation did not complete every finite episode")
        return [
            {
                "episode": i + 1,
                "seed": int(config.learning.best.seed),
                "world": i,
                "return": float(returns[i]),
                "length": int(lengths[i]),
                "event": Event(int(after.task.event[i])).name,
                "success": int(after.task.event[i]) == Event.SUCCESS,
                "flight_seconds": float(after.time[i]),
                **(
                    {
                        "position_rmse": float(
                            np.sqrt(after.task.error_sum[i] / max(1, after.task.samples[i]))
                        )
                    }
                    if env.task.name == "tracking"
                    else {}
                ),
                **({"gates_passed": int(after.task.gates[i])} if env.task.name == "racing" else {}),
            }
            for i in range(env.num_envs)
        ]

    return evaluate
