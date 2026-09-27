"""Verified frozen-policy storage and native inference reconstruction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import networks as ppo_networks

from drone_playground.evaluation.tracking import save_report, tree_digest
from drone_playground.learning.networks import network_factory

from .legacy import checkpoint_config


def save_policy(directory: Path, params, config: dict, step: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{int(step):010d}.pkl"
    temp = path.with_suffix(".tmp")
    model.save_params(str(temp), jax.tree.map(np.asarray, params))
    temp.replace(path)
    metadata = dict(
        config_version=2,
        step=int(step),
        config=checkpoint_config(config),
        observation_size=config.get("observation_size", 43),
        action_size=4,
        policy_family="brax",
        checkpoint_kind="inference-parameters",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameter_sha256=tree_digest(params),
        continuation={
            "ppo": "warm start only; optimizer/RNG reinitialized",
            "apg": "inference only; native APG has no restore hook",
            "shac": "full continuation is stored in training-state/",
            "dva": "full continuation is stored in training-state/",
        }[config["algorithm"]],
    )
    save_report(path.with_suffix(".json"), metadata)
    return path


def load_policy(path):
    path = Path(path).resolve()
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("policy_family") == "lotf_mlp":
        from drone_playground.learning.lotf_bptt import load_policy as load_lotf

        return load_lotf(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Checkpoint digest does not match its metadata")
    config = checkpoint_config(meta["config"])
    from drone_playground.composition import native_training_config

    native = native_training_config(config)
    preprocess = (
        running_statistics.normalize
        if native.get("normalize_observations", False)
        else types.identity_observation_preprocessor
    )
    network = network_factory(native)(
        meta["observation_size"], meta["action_size"], preprocess_observations_fn=preprocess
    )
    maker = (
        ppo_networks.make_inference_fn
        if native["algorithm"] in ("ppo", "dva")
        else apg_networks.make_inference_fn
    )
    params = model.load_params(str(path))
    if tree_digest(params) != meta["parameter_sha256"]:
        raise ValueError("Loaded parameter content changed")
    return (
        maker(network),
        params,
        {**meta, "config": config, "source_config_version": meta.get("config_version", 1)},
    )
