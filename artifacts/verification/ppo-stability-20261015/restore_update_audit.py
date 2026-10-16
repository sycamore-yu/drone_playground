"""Compare the production and diagnostic first update from the preserved Depth state."""

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint, ppo
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.records import atomic_json, source_identity

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[2] / "tools"))
from ppo_branch_diagnostic import tree_hash  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--deterministic-update", action="store_true")
parser.add_argument("--rehearsal", type=Path)
parser.add_argument("--rehearsal-weight", type=float, default=0.0)
parser.add_argument("--horizon", type=int)
parser.add_argument("--minibatches", type=int)
parser.add_argument("--gae-lambda", type=float)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

source = root / "depth-batch64-retention-pair/model_000300"
metadata = json.loads((source / "metadata.json").read_text())
config = OmegaConf.create(metadata["config"]["experiment"])
overrides = {
    key: value
    for key in ("horizon", "minibatches", "gae_lambda")
    if (value := getattr(args, key)) is not None
}
for key, value in overrides.items():
    config.learning.options[key] = value
trainer = Trainer(
    create_environment(config),
    kind=config.method.actor.kind,
    algorithm="ppo",
    seed=config.seed,
    config=OmegaConf.to_container(config.learning.options),
    actor_config=OmegaConf.to_container(config.method.actor),
    method=create_method(config),
    loss=config.learning.loss,
    backward_model=config.learning.backward_model,
    backward_options=OmegaConf.to_container(config.learning.backward_options),
)
compiler_options = (
    {"xla_gpu_exclude_nondeterministic_ops": True, "xla_gpu_autotune_level": 0}
    if args.deterministic_update
    else None
)
if compiler_options:
    trainer._update = jax.jit(
        lambda before, fraction: ppo.update(trainer, before, sampling_progress=fraction),
        compiler_options=compiler_options,
    )
state, _ = checkpoint.load_model_state(source, trainer.initialize())
before_hash = tree_hash(state)
rehearsal_loss, rehearsal_contract = None, None
if args.rehearsal is not None:
    from ppo_rehearsal_diagnostic import install_rehearsal

    rehearsal_loss, rehearsal_contract = install_rehearsal(
        trainer, state, args.rehearsal, args.rehearsal_weight
    )
progress = int(state.interactions) / config.learning.max_interactions
production, production_metrics = trainer.update(state, sampling_progress=progress)
jax.block_until_ready(production)
repeated, _ = trainer.update(state, sampling_progress=progress)
jax.block_until_ready(repeated)
diagnostic, diagnostic_metrics = jax.jit(
    lambda before, fraction: ppo.update(trainer, before, sampling_progress=fraction),
    compiler_options=compiler_options,
)(state.replace(updates=0, interactions=0), jnp.asarray(progress, jnp.float32))
diagnostic = diagnostic.replace(
    updates=state.updates + 1,
    interactions=state.interactions + trainer.config["horizon"] * trainer.env.num_envs,
)
assert tree_hash(state) == before_hash
differences = {}
for name in state.__dataclass_fields__:
    maxima = []
    for a, b in zip(
        jax.tree.leaves(getattr(production, name)),
        jax.tree.leaves(getattr(diagnostic, name)),
        strict=True,
    ):
        if hasattr(a, "dtype") and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        a, b = np.asarray(a), np.asarray(b)
        maxima.append(float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0)))
    differences[name] = max(maxima, default=0)
historical = next(
    record
    for line in (root / "depth-batch64-pilot.log").read_text().splitlines()
    if line.startswith("{") and (record := json.loads(line)).get("update") == int(state.updates) + 1
)
metrics = {
    key: {
        "historical": historical.get(key),
        "production": float(value),
        "diagnostic": float(diagnostic_metrics[key]),
    }
    for key, value in production_metrics.items()
    if key in diagnostic_metrics
}
result = {
    "source": str(source),
    "source_identity": source_identity(),
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
    "update_compiler_options": compiler_options,
    "learning_overrides": overrides,
    "rehearsal": rehearsal_contract,
    "restored_state_sha256": before_hash,
    "production_state_sha256": tree_hash(production),
    "diagnostic_state_sha256": tree_hash(diagnostic),
    "repeated_production_state_sha256": tree_hash(repeated),
    "same_compiled_update_exact": tree_hash(repeated) == tree_hash(production),
    "production_diagnostic_max_abs_difference": differences,
    "metrics": metrics,
    "scope": "One actual update, not long-run reproducibility. Historical metrics remain the "
    "original h32 reference, not matched samples when learning options are overridden.",
}
if rehearsal_loss is not None:
    result["rehearsal_kl"] = {
        name: float(rehearsal_loss({"params": value.params, "log_std": value.log_std}, state.rng))
        for name, value in [("before", state), ("after", production)]
    }
atomic_json(args.output, result)
print(json.dumps({key: value for key, value in result.items() if key != "metrics"}), flush=True)
trainer.env.sim.close()
