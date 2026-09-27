#!/usr/bin/env python3
"""Repeat only evaluations missing all-case traces, preserving the original runs.

The replacement rule is evidence-based (archive absent), never score-based.
Training budgets/checkpoints/scene seeds and numerical settings are unchanged.
Native asynchronous results may differ; both reports remain available.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def wait_completed(path):
    while True:
        result = read(path / "result.json")
        if result.get("status") == "completed":
            return
        if result.get("status") == "failed":
            raise RuntimeError(f"Original run failed: {path.name}")
        time.sleep(55)


def run_one(source_id):
    source = ROOT / "experiments" / source_id
    config = read(source / "manifest.json")["config"]
    target_id = source_id + "-archive-v1"
    config["run_id"] = target_id
    config["mode"] = "evaluate"
    if config["policy"]["name"] in ("native_ego", "native_super"):
        config["policy"]["port"] += 1000
    else:
        # The last original train can finish before its two GPU evaluations.
        # Wait for the complete learning queue, not just its training states.
        revision = source_id.split("-seed0-", 1)[1].rsplit("-", 1)[0]
        queue = ROOT / "experiments" / ("p5-matrix-" + revision) / "queue-state.json"
        while True:
            rows = read(queue) or []
            if any(row.get("status") == "failed" for row in rows):
                raise RuntimeError("Original learning queue has a failed unit")
            if len(rows) == 8 and all(row.get("status") == "completed" for row in rows):
                break
            time.sleep(55)
    os.environ.pop("PYTHONPATH", None)
    os.environ["JAX_PLATFORMS"] = "cpu" if config["training"]["device"] == "cpu" else "cuda"
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from drone_playground.composition import run_experiment

    run_experiment(config, ROOT, target_id)
    before = read(source / "eval/report.json")
    after = read(ROOT / "experiments" / target_id / "eval/report.json")
    fields = ("arrived", "collision", "out_of_bounds", "numerical_failure", "timeout")
    evidence = {
        "source_run": source_id, "replacement_run": target_id,
        "reason": "Original evaluation lacks all-episode frame archives; replacement predetermined without considering score",
        "training_interactions_added": 0,
        "same_scene_bank": before["scene_bank_sha256"] == after["scene_bank_sha256"],
        "same_parameter_digest": before.get("parameter_sha256") == after.get("parameter_sha256"),
        "original_outcomes": {k: before[k] for k in fields},
        "replacement_outcomes": {k: after[k] for k in fields},
    }
    if not evidence["same_scene_bank"] or not evidence["same_parameter_digest"]:
        raise RuntimeError("Archive replacement changed frozen input identity")
    (ROOT / "experiments" / target_id / "archive-repair.json").write_text(
        json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v2")
    parser.add_argument("--kind", choices=("learning", "native"))
    parser.add_argument("--one")
    args = parser.parse_args()
    if args.one:
        run_one(args.one)
        return
    if not args.kind:
        parser.error("--kind or --one is required")
    logs = ROOT / "experiments" / ("p5-archive-repairs-" + args.revision)
    logs.mkdir(parents=True, exist_ok=True)
    learned = [f"p5-formal-{t}-{s}-{a}-seed0-{args.revision}"
               for t in ("static", "dynamic") for s in ("depth", "lidar") for a in ("ppo", "dva")]
    if args.kind == "learning":
        # Do not compete with the frozen training queue for GPU memory/time.
        for base in learned:
            wait_completed(ROOT / "experiments" / base)
        bases = learned
    else:
        bases = [f"p5-formal-{t}-{m}-{args.revision}" for t in ("static", "dynamic")
                 for m in ("ego", "super")]

    def complete(base):
        for split in ("dev", "heldout"):
            source_id = base + "-" + split
            source = ROOT / "experiments" / source_id
            wait_completed(source)
            if (source / "traces/index.json").exists():
                continue
            target = ROOT / "experiments" / (source_id + "-archive-v1")
            if target.exists():
                if (read(target / "result.json").get("status") == "completed"
                        and (target / "archive-repair.json").exists()):
                    continue
                raise RuntimeError(f"Refusing to overwrite incomplete {target.name}")
            with (logs / (target.name + ".log")).open("w") as handle:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", source_id],
                               cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({"source": source_id, "archive_repair": target.name,
                              "status": "completed"}), flush=True)

    with ThreadPoolExecutor(max_workers=2 if args.kind == "native" else 1) as pool:
        list(pool.map(complete, bases))


if __name__ == "__main__":
    main()
