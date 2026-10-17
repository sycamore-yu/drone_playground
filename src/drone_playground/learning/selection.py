"""Episode-return measurement and optional within-run checkpoint selection."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning.objective import EpisodeObjective
from drone_playground.simulation.factory import create_environment
from drone_playground.simulation.tasks import Event


class EarlyStopping:
    """Stop the update loop on an evaluation target or a measured plateau."""

    def __init__(self, options, source, saved=None):
        """Keep recoverable evaluation history separate from policy selection."""
        self.defaults = dict(options)
        self.source = source
        self.state = dict(saved) if saved is not None else {}
        self.configure({})

    def configure(self, overrides):
        """Apply runtime rules; target/patience edits retain comparable history."""
        if not isinstance(overrides, dict) or overrides.keys() - self.defaults.keys():
            raise ValueError("Unknown early_stopping control fields")
        rules = {**self.defaults, **overrides}
        if rules["metric"] not in ("mean_reward", "success_rate", "position_rmse"):
            raise ValueError(
                "early_stopping.metric must be mean_reward, success_rate, or position_rmse"
            )
        if type(rules["patience"]) is not int or rules["patience"] < 0:
            raise ValueError("early_stopping.patience must be a nonnegative evaluation count")
        for name in ("target", "min_delta"):
            value = rules[name]
            if name == "target" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"early_stopping.{name} must be finite")
        if rules["min_delta"] < 0:
            raise ValueError("early_stopping.min_delta must be nonnegative")
        active = rules["target"] is not None or rules["patience"] > 0
        if active and self.source is None:
            raise ValueError("Early stopping requires a Best measurement source")
        if active and rules["metric"] != "mean_reward" and self.source != "evaluation":
            raise ValueError("Quality-metric early stopping requires fixed evaluation")
        if (
            rules["metric"] == "success_rate"
            and rules["target"] is not None
            and not 0 <= rules["target"] <= 1
        ):
            raise ValueError("early_stopping success_rate target must be in [0, 1]")
        if (
            rules["metric"] == "position_rmse"
            and rules["target"] is not None
            and rules["target"] < 0
        ):
            raise ValueError("early_stopping position_rmse target must be nonnegative")
        previous = self.state.get("options")
        changed = previous != rules
        if previous is None or any(previous[key] != rules[key] for key in ("metric", "min_delta")):
            self.state.update(best_score=None, last_score=None, bad_evaluations=0, evaluations=0)
        self.state["options"] = rules
        return changed

    def observe(self, measurement):
        """Observe one completed selection evaluation without advancing training."""
        rules = self.state["options"]
        rows = measurement["episode_records"]
        metric = rules["metric"]
        if metric != "mean_reward" and not rows:
            raise ValueError("Quality-metric early stopping requires episode records")
        if metric == "mean_reward":
            score = measurement["mean_reward"]
        elif metric == "success_rate":
            score = sum(row["success"] for row in rows) / len(rows)
        else:
            if any("position_rmse" not in row for row in rows):
                raise ValueError("position_rmse early stopping requires Tracking episode metrics")
            score = sum(row["position_rmse"] for row in rows) / len(rows)
        if not math.isfinite(score):
            raise FloatingPointError("Early stopping received a nonfinite metric")
        best = self.state["best_score"]
        direction = -1 if metric == "position_rmse" else 1
        improved = best is None or direction * score > direction * best + rules["min_delta"]
        self.state.update(
            last_score=score,
            best_score=score if improved else best,
            bad_evaluations=0 if improved else self.state["bad_evaluations"] + 1,
            evaluations=self.state["evaluations"] + 1,
            all_successful=all(row.get("success", False) for row in rows) if rows else False,
            confirmed=measurement.get("confirmed", True),
        )

    def reason(self):
        """Check the most recent evaluation, including after a live rule change."""
        score = self.state["last_score"]
        if score is None:
            return None
        rules = self.state["options"]
        direction = -1 if rules["metric"] == "position_rmse" else 1
        target = rules["target"]
        if (
            target is not None
            and self.state.get("confirmed", True)
            and direction * (score - target) >= 0
            and (rules["metric"] != "position_rmse" or self.state["all_successful"])
        ):
            reason = "target_reached"
        elif rules["patience"] and self.state["bad_evaluations"] >= rules["patience"]:
            reason = "no_improvement"
        else:
            return None
        return {
            "reason": reason,
            "metric": rules["metric"],
            "value": score,
            "target": target,
            "bad_evaluations": self.state["bad_evaluations"],
            "patience": rules["patience"],
            "min_delta": rules["min_delta"],
        }


class RewardSelection:
    """Compare completed episode returns without changing the training stream."""

    def __init__(self, config, trainer, saved=None):
        """Bind the selection recipe and restore its exact measurement history."""
        self.config = config.learning.best
        self.metric = self.config.get("metric", "mean_reward")
        self.confirmation_episodes = self.config.get("confirmation_episodes", 0)
        if self.metric not in {"mean_reward", "success_rate"}:
            raise ValueError("learning.best.metric must be mean_reward or success_rate")
        if type(self.confirmation_episodes) is not int or self.confirmation_episodes < 0:
            raise ValueError("learning.best.confirmation_episodes must be a nonnegative integer")
        if (
            self.metric == "success_rate" or self.confirmation_episodes
        ) and self.config.source == "training":
            raise ValueError("Success selection and confirmation require evaluation measurements")
        if self.confirmation_episodes and self.confirmation_episodes < self.config.episodes:
            raise ValueError(
                "Confirmation must contain at least as many episodes as the quick check"
            )
        self._trainer = trainer
        self._full_config = config
        self.confirm = None
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
            quick_episodes = len(rows)
            quick_successes = sum(row.get("success", False) for row in rows)
            confirmed = not self.confirmation_episodes
            if self.confirmation_episodes and quick_successes == quick_episodes:
                if self.confirm is None:
                    config = OmegaConf.create(
                        OmegaConf.to_container(self._full_config, resolve=True)
                    )
                    config.learning.best.episodes = self.confirmation_episodes
                    config.learning.best.seed = (config.learning.best.seed + 7919) % 2**32
                    self.confirm = fixed_evaluator(config, self._trainer)
                rows = self.confirm(state.params)
                confirmed = True
            mean = float(np.mean([row["return"] for row in rows]))
            count = len(rows)
        else:
            count = self.state["window_episodes"]
            if not count:
                return None
            rows = []
            mean = self.state["window_return"] / count
            confirmed, quick_episodes, quick_successes = True, count, None
        if not math.isfinite(mean):
            raise FloatingPointError("Checkpoint selection produced a nonfinite episode return")
        self.state.update(
            last_interactions=int(state.interactions), window_episodes=0, window_return=0.0
        )
        previous = self.state["best_mean_reward"]
        if self.metric == "success_rate":
            rate = sum(row["success"] for row in rows) / len(rows)
            score = [int(confirmed), rate, mean]
            previous_score = self.state.get("best_score")
            improved = previous_score is None or tuple(score) > tuple(previous_score)
        else:
            score = [mean]
            improved = previous is None or mean > previous
        return {
            "source": self.config.source,
            "mean_reward": mean,
            "episodes": count,
            "episode_records": rows,
            "improved": improved,
            "selection_score": score,
            "confirmed": confirmed,
            "quick_episodes": quick_episodes,
            "quick_successes": quick_successes,
        }


def fixed_evaluator(config, trainer):
    """Build a deterministic, separate Simulation batch with fixed seeds and difficulty."""
    settings = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    settings.simulation.num_envs = config.learning.best.episodes
    if "procedural" in settings.simulation:
        settings.simulation.procedural.training_seed = config.learning.best.seed
        if settings.simulation.procedural.get("layout") == "matched":
            settings.simulation.procedural.start_sampling = "distance_stratified"
    env = create_environment(settings)
    objective = EpisodeObjective(
        env, trainer.config, recipe=trainer.objective.recipe, loss_name=trainer.loss_name
    )
    initial = env.reset(
        jax.random.PRNGKey(config.learning.best.seed),
        **trainer._reset_options,
        **({"sampling_progress": 1.0} if trainer.distance_curriculum else {}),
    )
    distance = (
        np.asarray(
            jnp.linalg.norm(env.goal_for(initial) - initial.physics.states.pos[:, 0], axis=-1)
        )
        if env.task.name == "navigation"
        else None
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
                        "start_goal_distance_m": float(distance[i]),
                        "distance_group": "short"
                        if distance[i] < 30
                        else "medium"
                        if distance[i] < 80
                        else "long",
                    }
                    if distance is not None
                    else {}
                ),
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
