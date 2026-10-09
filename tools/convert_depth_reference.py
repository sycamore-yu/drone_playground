"""Extract one declared research checkpoint as a fresh-initialization candidate.

Run from the repository root with ``pixi run python tools/convert_depth_reference.py``.
This copies tensors only; point-mass training results do not establish Crazyflow quality.
"""

import argparse
import hashlib
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from drone_playground.learning.checkpoint import save_inference
from drone_playground.simulation.networks import Actor

CHECKSUM = "2c9f0d540095c77391772cdd34aa1d795d3e0cb6178258ef0e36933fbaecbcba"


class ReferenceState:
    """Inert recipient for the declared archive's attributes; no historical code imports."""


class ReferenceReader(pickle.Unpickler):
    """Load the declared reference state without importing historical project code."""

    def find_class(self, module, name):
        """Resolve the allowed historical state as an inert parameter container."""
        if module.startswith("drone_playground."):
            if (
                module == "drone_playground.learning.algorithms.pointcloud_bptt"
                and name == "TrainingState"
            ):
                return ReferenceState
            raise ValueError(f"Unexpected historical class {module}.{name}")
        return super().find_class(module, name)


def main():
    """Convert the declared Depth weights into an explicit initialization archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/reference_initialization/depth.policy.zip")
    )
    args = parser.parse_args()
    if not np.isfinite(args.action_gain) or args.action_gain <= 0:
        parser.error("--action-gain must be finite and positive")
    path = Path("research/checkpoints/weights") / f"{CHECKSUM}.pkl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CHECKSUM
    with path.open("rb") as stream:
        reference = ReferenceReader(stream).load().params["params"]
    converted = {
        **{f"conv_{i}": reference[f"convolutions_{i}"] for i in range(3)},
        "perception_projection": reference["image_projection"],
        "state_projection": reference["state_projection"],
        "gru": reference["memory"],
        "action_head": {
            "kernel": reference["output_projection"]["kernel"][:, :3] * args.action_gain
        },
        "velocity_head": {"kernel": reference["output_projection"]["kernel"][:, 3:]},
    }
    parameters = jax.tree.map(jnp.asarray, {"params": converted})
    actor = Actor("depth")
    observation = {"state": jnp.zeros((1, 10)), "depth": jnp.zeros((1, 12, 16, 1))}
    memory = jnp.zeros((1, 192))
    template = actor.init(jax.random.PRNGKey(0), observation, memory)
    assert jax.tree.structure(parameters) == jax.tree.structure(template)
    for source, target in zip(jax.tree.leaves(parameters), jax.tree.leaves(template), strict=True):
        assert source.shape == target.shape and source.dtype == target.dtype
        assert np.isfinite(source).all()
    assert all(np.isfinite(value).all() for value in actor.apply(parameters, observation, memory))
    with initialize_config_module(version_base=None, config_module="drone_playground.configs"):
        config = compose(config_name="config", overrides=["experiment=navigation_depth"])
    save_inference(
        args.output,
        parameters,
        kind="depth",
        config={"experiment": OmegaConf.to_container(config, resolve=True)},
        provenance={
            "source_archive": str(path),
            "source_sha256": CHECKSUM,
            "transfer": "Rename tensors; split output into action and velocity heads",
            "source_physics": "historical point_mass_lag; not Crazyflow",
            "initial_action_head_scale": args.action_gain,
            "validation": "Shapes and finite Actor execution; physical evaluation required",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
