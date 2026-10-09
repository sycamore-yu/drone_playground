"""Convert the declared LiDAR reference into an approximate initialization candidate.

Run with ``pixi run python tools/convert_lidar_reference.py``. This transfers weights;
the reference's point-mass result is not a Crazyflow benchmark result.
"""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from convert_depth_reference import ReferenceReader
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from drone_playground.learning.checkpoint import save_inference
from drone_playground.simulation.networks import Actor

CHECKSUM = "e1750add4300262fe2907943651e6530c4ba27667751b4a5cba98cb7b1b1f9d7"


def main():
    """Convert the declared LiDAR weights into an explicit initialization archive."""
    path = Path("research/checkpoints/weights") / f"{CHECKSUM}.pkl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError("Reference archive checksum mismatch")
    with path.open("rb") as stream:
        reference = ReferenceReader(stream).load().params["params"]
    converted = {
        **{f"point_{i}": dict(reference[f"point_{i}"]) for i in range(3)},
        "perception_projection": {"kernel": reference["point_projection"]["kernel"]},
        "state_projection": {
            "kernel": reference["state_projection"]["kernel"],
            "bias": reference["state_projection"]["bias"] + reference["point_projection"]["bias"],
        },
        "gru": reference["memory"],
        "action_head": {"kernel": reference["acceleration"]["kernel"] / 6.0},
    }
    # Fold the reference's 0.1 point-input scale into its first affine layer.
    converted["point_0"]["kernel"] *= 0.1
    parameters = jax.tree.map(jnp.asarray, {"params": converted})
    actor = Actor("lidar")
    observation = {
        "state": jnp.zeros((1, 10)),
        "points": jnp.zeros((1, 1024, 3)),
        "mask": jnp.ones((1, 1024), bool),
    }
    memory = jnp.zeros((1, 192))
    template = actor.init(jax.random.PRNGKey(0), observation, memory)
    if jax.tree.structure(parameters) != jax.tree.structure(template):
        raise ValueError("Reference and Actor parameter structures disagree")
    for source, target in zip(jax.tree.leaves(parameters), jax.tree.leaves(template), strict=True):
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError("Reference parameter shape or dtype mismatch")
        if not np.isfinite(source).all():
            raise ValueError("Reference contains nonfinite parameters")
    if not all(np.isfinite(value).all() for value in actor.apply(parameters, observation, memory)):
        raise ValueError("Converted policy produced nonfinite outputs")
    with initialize_config_module(version_base=None, config_module="drone_playground.configs"):
        config = compose(config_name="config", overrides=["experiment=navigation_lidar"])
    output = Path("results/reference_initialization/lidar.policy.zip")
    save_inference(
        output,
        parameters,
        kind="lidar",
        config={"experiment": OmegaConf.to_container(config, resolve=True)},
        provenance={
            "source_archive": str(path),
            "source_sha256": CHECKSUM,
            "source_physics": "historical point_mass_lag; not Crazyflow",
            "point_scale_folded_into_first_kernel": 0.1,
            "projection_bias": "Added to the state projection bias before shared activation",
            "action_kernel_gain": 1 / 6,
            "omitted_action_bias": np.asarray(reference["acceleration"]["bias"]).tolist(),
            "transfer": "Approximate initialization; current Actor has a bias-free tanh head",
            "validation": "Shapes and finite Actor execution; physical evaluation required",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
