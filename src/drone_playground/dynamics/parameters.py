"""Physical parameter ranges shared by the Crazyflow and LOTF containers."""

import math

import jax
import jax.numpy as jnp


def parameter_ranges(randomization, overrides=None):
    """Validate multiplicative factors without constructing a dynamics model."""
    settings = {"enabled": False} if randomization is None else randomization
    if not isinstance(settings, dict) or set(settings) - {"enabled", "dynamics"}:
        raise ValueError("Unsupported domain_randomization fields")
    if not isinstance(settings.get("enabled"), bool):
        raise ValueError("domain_randomization.enabled must be an explicit boolean")
    allowed = {"mass", "inertia", "motor_strength", "drag"}
    declared = settings.get("dynamics", {})
    if not isinstance(declared, dict) or set(declared) - allowed:
        raise ValueError("Unknown dynamics randomization fields")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("parameter_overrides must be a mapping")
    values = dict(declared) if settings["enabled"] else {}
    values.update({name: [value, value] for name, value in (overrides or {}).items()})
    if set(values) - allowed:
        raise ValueError(f"Unknown dynamics randomization fields: {sorted(set(values) - allowed)}")
    ranges = {}
    for name, bounds in values.items():
        if bounds is None:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"domain_randomization.dynamics.{name} requires [low, high]")
        low, high = map(float, bounds)
        if not all(map(math.isfinite, (low, high))) or low <= 0 or high < low:
            raise ValueError(
                f"domain_randomization.dynamics.{name} requires finite 0 < low <= high"
            )
        ranges[name] = (low, high)
    if settings["enabled"] and not ranges:
        raise ValueError("Enabled domain_randomization requires physical parameter ranges")
    return ranges


def randomize_parameters(data, key, ranges):
    """Sample the declared factors in their original order at one reset."""
    if not ranges:
        return data
    keys = iter(jax.random.split(key, len(ranges)))
    params = data.params
    for name, (low, high) in ranges.items():
        factor = jax.random.uniform(next(keys), (), minval=low, maxval=high)
        if name == "mass":
            params = params.replace(mass=params.mass * factor)
        elif name == "inertia":
            inertia = params.J * factor
            params = params.replace(J=inertia, J_inv=jnp.linalg.inv(inertia))
        elif name == "motor_strength":
            if hasattr(params, "cmd_f_coef"):
                params = params.replace(cmd_f_coef=params.cmd_f_coef * factor)
            else:
                values = {"rpm2thrust": params.rpm2thrust * factor}
                if hasattr(params, "rpm2torque"):
                    values["rpm2torque"] = params.rpm2torque * factor
                params = params.replace(**values)
        elif name == "drag":
            params = params.replace(drag_matrix=params.drag_matrix * factor)
    return data.replace(params=params)


def physical_parameters(data):
    """Return the parameters actually carried by the current physical state."""
    parameters = {"mass_kg": data.params.mass, "inertia_kg_m2": data.params.J}
    for name in ("cmd_f_coef", "rpm2thrust", "rpm2torque", "drag_matrix"):
        if hasattr(data.params, name):
            parameters[name] = getattr(data.params, name)
    return parameters
