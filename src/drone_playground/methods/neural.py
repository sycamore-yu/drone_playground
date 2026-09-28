"""Frozen neural policies expose the same observation-to-command role."""

from dataclasses import dataclass

import jax

from drone_playground.runs.checkpoints import load_policy


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
