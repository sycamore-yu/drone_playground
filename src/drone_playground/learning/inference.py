"""Frozen policy inference at a physical reference or setpoint boundary."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from drone_playground.references import Trajectory
from drone_playground.runtime.decision import Decision, output_reply


class FrozenNeuralCommand:
    """Reuse a frozen policy with a recorded physical output and input semantics."""

    input_kind, derivatives = None, "none"

    def __init__(self, checkpoint, frequency_hz=None, input_kind=None):
        self.checkpoint = checkpoint
        self.frequency_hz = frequency_hz
        self.input_kind = input_kind

    def bind(self, env, directory):
        del directory
        checkpoint, frequency_hz, input_kind = self.checkpoint, self.frequency_hz, self.input_kind
        import jax

        self.policy, self.env = NeuralPolicy.load(checkpoint), env
        if input_kind not in (None, "trajectory"):
            raise ValueError("Frozen neural upstream input must be an explicitly timed trajectory")
        self.input_kind = input_kind
        self.frequency = float(env.freq if frequency_hz is None else frequency_hz)
        metadata = self.policy.metadata
        self.output_kind = metadata["config"]["method"]["output"]
        self.goal_source = metadata["config"]["method"].get("goal_source", "task_goal")
        if self.goal_source not in ("task_goal", "observation_reference"):
            raise ValueError("Frozen neural policy has an unknown goal source")
        self.decoder = None
        if metadata.get("physical_decoder"):
            from drone_playground.control.decoders import PhysicalActionDecoder

            self.decoder = PhysicalActionDecoder(**metadata["physical_decoder"])
            if (
                self.decoder.kind != self.output_kind
                or self.decoder.action_size != metadata["action_size"]
            ):
                raise ValueError(
                    "Frozen neural physical decoder differs from checkpoint dimensions or type"
                )
        elif self.output_kind in ("waypoint", "trajectory"):
            raise ValueError("Frozen neural geometric output requires an explicit physical decoder")
        if (self.decoder is None and self.output_kind != env.controller.input_kind) or (
            input_kind is None and metadata["observation_size"] != env.observation_size
        ):
            raise ValueError(
                "Frozen neural observation/command decoder differs from this environment"
            )
        source = metadata["config"]["env"]
        from drone_playground.environments.factory import build_observer, build_sensor

        self.observer = observer = build_observer(
            metadata["config"], build_sensor(metadata["config"])
        )
        # Equal vector lengths alone do not establish equal field meanings.
        if input_kind is None and (
            type(observer) is not type(env.task.observation)
            or vars(observer) != vars(env.task.observation)
        ):
            raise ValueError("Frozen neural observation field semantics differ")
        if self.goal_source == "observation_reference" and observer.name != "state_reference":
            raise ValueError("Frozen geometric goal requires the recorded reference observation")
        dynamics = source["dynamics"]
        if (
            source["freq"] != self.frequency
            or dynamics["drone"] != env.drone
            or dynamics["forward"] != env.dynamics.forward
        ):
            raise ValueError("Frozen neural clock/model action decoder differs")
        if input_kind == "trajectory":
            from drone_playground.environments.observations.state import TrackingObservation

            if (
                type(observer) is not TrackingObservation
                or observer.name != "state_reference"
                or observer.n_samples < 1
                or metadata["observation_size"] != observer.size
            ):
                raise ValueError(
                    "Upstream trajectory requires a recorded state_reference observation"
                )
            stride = self.frequency * observer.interval
            # The original tracking/racing encoders round fractional strides
            # differently. Require an unambiguous physical sampling clock.
            if (
                not np.isfinite(stride)
                or stride <= 0
                or not np.isclose(stride, round(stride), atol=1e-8, rtol=0)
            ):
                raise ValueError(
                    "Upstream reference interval must cover an integer number of policy ticks"
                )
            self.reference_offsets = np.arange(observer.n_samples) * round(stride) / self.frequency
        self.provenance = dict(
            checkpoint=str(Path(checkpoint).resolve()),
            sha256=metadata["sha256"],
            step=metadata["step"],
            observation=source["task"]["observation"],
            output=self.output_kind,
        )
        self.provenance["goal_source"] = self.goal_source
        if input_kind is not None:
            self.provenance["reference_source"] = "upstream_trajectory"
            self.provenance["reference_offsets_seconds"] = self.reference_offsets.tolist()
        if self.decoder is not None:
            self.provenance["physical_decoder"] = metadata["physical_decoder"]
            self.decode = jax.jit(self.decoder.decode)
        self.infer = jax.jit(self.policy.act)
        shape = jax.eval_shape(
            self.policy.act,
            jax.ShapeDtypeStruct((metadata["observation_size"],), np.float32),
        )
        if shape.shape != (metadata["action_size"],):
            raise ValueError("Frozen neural parameter output differs from checkpoint dimensions")
        self.count = 0
        return self

    def start(self, calibration, goal, limits, task):
        self.count = 0
        self.goal = goal

    def step(self, packet, upstream):
        import jax
        import jax.numpy as jnp

        self.count += 1
        if self.input_kind == "trajectory":
            if not isinstance(upstream, Trajectory):
                raise ValueError("Frozen neural tracker requires upstream Trajectory")
            times = packet["time"] + self.reference_offsets
            if times[0] < upstream.start_time - 1e-9 or times[-1] > upstream.end_time + 1e-9:
                return Decision(
                    status="no_plan",
                    output=None,
                    plan_id=str(self.count),
                    generated_at=float(packet["time"]),
                    valid_until=float(packet["time"]),
                    diagnostics={
                        "reason": "insufficient_reference_horizon",
                        "required_reference_until": float(times[-1]),
                    },
                )
            reference = upstream.sample_many(times)
            fields = dict(
                pos="position",
                quat="quaternion",
                vel="velocity",
                ang_vel="angular_velocity",
            )
            states = SimpleNamespace(
                **{field: jnp.asarray(packet[key])[None, None] for field, key in fields.items()}
            )
            observation = self.observer(states, jnp.asarray(reference["position"]))
        else:
            observation = packet["policy_observation"]
            observation = (
                jax.tree.map(jnp.asarray, observation)
                if isinstance(observation, dict)
                else jnp.asarray(observation)
            )
        action = self.infer(observation)
        if "goal" in packet:
            self.goal = packet["goal"]
        if self.decoder is None:
            value = self.env.controller.setpoint(self.env.physical_action(action))
        else:
            goal = (
                self.observer.reference_goal(observation)
                if self.goal_source == "observation_reference"
                else self.goal
            )
            decoded = self.decode(action, packet["position"], packet["velocity"], goal)
            value = self.decoder.message(decoded, packet["time"])
        return output_reply(
            value,
            packet["time"],
            str(self.count),
            packet["time"] + 1 / self.frequency,
        )

    def close(self):
        pass


@dataclass
class NeuralPolicy:
    make_policy: object
    parameters: object
    metadata: dict

    @classmethod
    def load(cls, path):

        return cls(*load_policy(path))

    def act(self, observation):
        return self.make_policy(self.parameters, deterministic=True)(
            observation, jax.random.PRNGKey(0)
        )[0]


def load_policy(path):
    """Rebuild inference from verified parameters and the saved environment contract."""
    from brax.training import types
    from brax.training.acme import running_statistics
    from brax.training.agents.apg import networks as apg_networks
    from brax.training.agents.ppo import networks as ppo_networks

    from drone_playground.artifacts.checkpoints import load_checkpoint
    from drone_playground.artifacts.reporting import tree_digest
    from drone_playground.artifacts.schema import require_current
    from drone_playground.networks.factory import network_factory

    params, meta = load_checkpoint(path)
    config = require_current(meta["config"])
    from drone_playground.environments.factory import build_observer, build_sensor

    observer = build_observer(config, build_sensor(config))
    if (
        observer.specification() != meta["observation_spec"]
        or observer.size != meta["observation_size"]
    ):
        raise ValueError(
            "Saved observation fields or dimensions differ from the reconstructed component"
        )
    from drone_playground.learning.brax_configuration import native_training_config

    native = native_training_config(config)
    preprocess = (
        running_statistics.normalize
        if native.get("normalize_observations", False)
        else types.identity_observation_preprocessor
    )
    network = network_factory(native)(
        meta["observation_size"],
        meta["action_size"],
        preprocess_observations_fn=preprocess,
    )
    maker = (
        ppo_networks.make_inference_fn
        if native["algorithm"] in ("ppo", "dva")
        else apg_networks.make_inference_fn
    )
    if tree_digest(params) != meta["parameter_sha256"]:
        raise ValueError("Loaded parameter content changed")
    return (
        maker(network),
        params,
        {**meta, "config": config},
    )
