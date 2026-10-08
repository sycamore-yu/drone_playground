"""Runtime forces and moments evaluated on the native physics clock."""

import jax
import jax.numpy as jnp


def external_wrench(data):
    """Sample runtime forces on the actual physics clock, independently of DR."""
    plugins = data.plugins
    time = data.core.steps[0, 0] / data.core.freq
    index = jnp.floor(time / plugins["gust_period_s"]).astype(jnp.int32)
    gaussian, uniform = jax.random.split(jax.random.fold_in(plugins["disturbance_key"], index))
    force = (
        plugins["external_force_world_n"]
        + jax.random.normal(gaussian, (3,)) * plugins["gust_std_n"]
        + jax.random.uniform(uniform, (3,), minval=-1, maxval=1)
        * plugins["force_uniform_half_width_n"]
    )
    return force, plugins["external_torque_body_nm"]
