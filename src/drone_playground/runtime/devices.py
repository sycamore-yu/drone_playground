"""Keep physical execution placement independent from an optimizer backend."""

from contextlib import contextmanager

import jax


@contextmanager
def execution_scope(device):
    """Select the execution device context without modifying global configuration."""
    if device not in ("cpu", "gpu"):
        raise ValueError("Runtime device must be cpu or gpu")
    with jax.default_device(jax.devices(device)[0]):
        yield
