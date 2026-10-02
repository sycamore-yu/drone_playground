"""Frozen neural policy execution; construction belongs to networks.factory."""

from dataclasses import dataclass

import jax


@dataclass
class NeuralPolicy:
    make_policy: object
    parameters: object
    metadata: dict

    @classmethod
    def load(cls, path):
        from drone_playground.artifacts.checkpoints import load_policy

        return cls(*load_policy(path))

    def act(self, observation):
        return self.make_policy(self.parameters, deterministic=True)(
            observation, jax.random.PRNGKey(0)
        )[0]
