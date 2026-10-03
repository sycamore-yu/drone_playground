"""Verified frozen-policy storage and native inference reconstruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import networks as ppo_networks

from drone_playground.artifacts.layout import resolve_artifact
from drone_playground.artifacts.reporting import save_report, tree_digest
from drone_playground.artifacts.schema import require_current
from drone_playground.networks.factory import network_factory


def require_matching_physical_decoder(metadata, config):
    """Parameter transfer cannot silently reinterpret a geometric head's units."""
    from drone_playground.control.decoders import PhysicalActionDecoder

    current = require_current(config)
    expected = current["method"].get("physical_decoder")
    recorded = metadata.get("physical_decoder")
    if expected is None and recorded is None:
        return
    if (
        expected is None
        or recorded is None
        or PhysicalActionDecoder(**expected) != PhysicalActionDecoder(**recorded)
        or current["method"].get("goal_source", "task_goal")
        != metadata["config"]["method"].get("goal_source", "task_goal")
    ):
        raise ValueError(
            "Geometric warm start changes the physical decoder or goal source; explicit migration required"
        )


def save_policy(directory: Path, params, config: dict, step: int, *, physical_decoder=None) -> Path:
    current = require_current(config)
    component_only = (
        physical_decoder is not None and current["method"].get("physical_decoder") is None
    )
    if physical_decoder is None:
        physical_decoder = current["method"].get("physical_decoder")
    from drone_playground.environments.environment import (
        build_controller,
        build_observer,
        build_sensor,
    )

    controller = build_controller(current["env"]["controller"])
    observer = build_observer(current, build_sensor(current))
    action_size = len(controller.input_fields)
    if physical_decoder is not None:
        from drone_playground.control.decoders import PhysicalActionDecoder

        decoder = PhysicalActionDecoder(**physical_decoder)
        if decoder.kind != current["method"]["output"]:
            raise ValueError("Checkpoint method output differs from its physical decoder")
        action_size = decoder.action_size
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{int(step):010d}.pkl"
    temp = path.with_suffix(".tmp")
    model.save_params(str(temp), jax.tree.map(np.asarray, params))
    temp.replace(path)
    metadata = dict(
        config_version=4,
        step=int(step),
        config=current,
        observation_size=observer.size,
        observation_spec=observer.specification(),
        action_size=action_size,
        policy_family="brax",
        checkpoint_kind="component-inference-parameters"
        if component_only
        else "inference-parameters",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameter_sha256=tree_digest(params),
        continuation={
            "ppo": "warm start only; optimizer/RNG reinitialized",
            "apg": "inference only; native APG has no restore hook",
            "shac": "full continuation is stored in training-state/",
            "bptt": "full continuation is stored in training-state/",
            "dva": "full continuation is stored in training-state/",
        }[current["algorithm"]["name"]],
    )
    if physical_decoder is not None:
        # Persist every physical default so future decoder defaults cannot
        # silently change a frozen policy's units, anchors or finite horizon.
        metadata["physical_decoder"] = asdict(decoder)
        if component_only:
            metadata["continuation"] = (
                "frozen physical component; optimizer continuation is not encoded"
            )
    save_report(path.with_suffix(".json"), metadata)
    return path


def load_policy(path):
    path = resolve_artifact(path).resolve()
    meta = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Checkpoint digest does not match its metadata")
    config = require_current(meta["config"])
    from drone_playground.environments.environment import build_observer, build_sensor

    observer = build_observer(config, build_sensor(config))
    if (
        observer.specification() != meta["observation_spec"]
        or observer.size != meta["observation_size"]
    ):
        raise ValueError(
            "Saved observation fields or dimensions differ from the reconstructed component"
        )
    from drone_playground.learning.brax_configuration import native_training_config

    native = native_training_config(config)
    preprocess = (
        running_statistics.normalize
        if native.get("normalize_observations", False)
        else types.identity_observation_preprocessor
    )
    network = network_factory(native)(
        meta["observation_size"],
        meta["action_size"],
        preprocess_observations_fn=preprocess,
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
        {**meta, "config": config},
    )
