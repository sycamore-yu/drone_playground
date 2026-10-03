"""Crazyflow-based drone learning and control experiments."""

import os

# Crazyflow requires this setting before the first SciPy import.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

__version__ = "0.3.0"


def load(env_name, *, overrides=(), role="eval", count=1):
    """Load one Hydra environment preset without selecting a learning method."""
    from drone_playground.environments.environment import load as load_environment

    return load_environment(env_name, overrides=overrides, role=role, count=count)
