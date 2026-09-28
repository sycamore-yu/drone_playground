"""Frozen numerical recipes used ONLY to compare the pre-refactor algorithms.

Historical recipe labels are test case names. Production entry points compose
method/env with Hydra; their behavior is checked independently in v3 tests.
"""

import json
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

FIXTURES = Path(__file__).parent / "fixtures/architecture-v3-recipes.json"


def _historical_override(value):
    # Preserve the exact intent of original regression cases while the current
    # system consumes only the v3 ownership paths.
    mapping = (
        ("policy.planning.", "env.task.reference_generator."),
        ("policy.name", "method.implementation"),
        ("observation.sensor.", "env.sensor."),
        ("dynamics.backward", "algorithm.gradient.transition"),
        ("dynamics.decay_rate", "algorithm.gradient.decay_rate"),
        ("training.device", "runtime.device"),
        ("policy.", "method."),
        ("controller.", "env.execution.controller."),
        ("dynamics.", "env.execution.dynamics."),
        ("task.", "env.task."),
        ("scene.", "env.scene."),
        ("observation.", "env.observation."),
    )
    for old, new in mapping:
        if value.lstrip("+").startswith(old):
            value = value.replace(old, new, 1)
            break
    if value == "mode=evaluate":
        value = "mode=eval"
    if value == "mode=simulate":
        value = "mode=play"
    if value == "evaluation.environment=experiment":
        value = "evaluation.environment=config"
    return value


def compose_reference(label, overrides=None):
    values = json.loads(FIXTURES.read_text())
    config = OmegaConf.create(deepcopy(values[label]))
    OmegaConf.set_struct(config, True)
    for override in overrides or []:
        value = _historical_override(override)
        key, literal = value.lstrip("+").split("=", 1)
        parsed = OmegaConf.from_dotlist(["value=" + literal]).value
        OmegaConf.update(config, key, parsed, merge=True, force_add=value.startswith("+"))
    return OmegaConf.to_container(config, resolve=True)
