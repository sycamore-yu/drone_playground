"""Stable run locations, independent of experiment semantics encoded in names."""

import re
from pathlib import Path

_RESERVED = {".", "..", "runs", "selected", "scratch", "tmp", "main_result"}


def _run_id(run_id):
    if not run_id or Path(run_id).name != run_id or run_id in _RESERVED:
        raise ValueError("Run identity must be one directory name")


def _slug(value, fallback):
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text).strip("-._")
    return text or fallback


def experiment_scope(config):
    """Return the stable task/method grouping for one run."""
    config = (config or {}).get("components", config or {})
    env = config.get("env") or {}
    task = env.get("task") or config.get("task") or {}
    task_name = task if isinstance(task, str) else task.get("name")
    method = config.get("method") or {}
    method_name = method if isinstance(method, str) else method.get("name")
    if method_name == "policy":
        method_name = (config.get("algorithm") or {}).get("name")
    method_name = method_name or (config.get("algorithm") or {}).get("name")
    return _slug(task_name, "misc"), _slug(method_name, "run")


def iter_experiments(root):
    """Enumerate existing experiment run directories in stable order."""
    experiments = Path(root) / "results" / "runs"
    return sorted(p for p in experiments.glob("*/*/*") if p.is_dir())


def find_experiment(root, run_id):
    """Resolve a unique run identifier to its experiment directory."""
    _run_id(run_id)
    experiments = Path(root) / "results" / "runs"
    candidates = experiments.glob("*/*/" + run_id)
    existing = {p.resolve(): p for p in candidates if p.is_dir()}
    if len(existing) > 1:
        raise ValueError(f"Ambiguous run identity: {run_id}")
    return next(iter(existing.values()), None)


def experiment_directory(root, run_id, config=None):
    """Return an existing run or the scoped location for a new immutable run."""
    _run_id(run_id)
    existing = find_experiment(root, run_id)
    if existing is not None:
        return existing
    if config is None:
        raise ValueError("Creating a new run requires its Task/Method configuration")
    task, method = experiment_scope(config)
    return Path(root) / "results" / "runs" / task / method / run_id
