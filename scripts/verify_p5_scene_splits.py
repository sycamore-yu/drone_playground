#!/usr/bin/env python3
"""Audit per-instance geometry/motion identity across frozen P5 dataset splits."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from hydra.utils import instantiate

from drone_playground.composition import SPLIT_SEEDS
from drone_playground.tasks.scenes.navigation import make_bank

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v2")
    args = parser.parse_args()
    target = ROOT / "docs/verification" / ("p5-results-" + args.revision)
    raw = ROOT / "experiments" / ("p5-scene-audit-" + args.revision)
    raw.mkdir(parents=True, exist_ok=True)
    records, intersections, sources = [], [], []
    issues = []
    for task in ("static", "dynamic"):
        configs = []
        for sensor, method in itertools.product(("depth", "lidar"), ("ppo", "dva")):
            run_id = f"p5-formal-{task}-{sensor}-{method}-seed0-{args.revision}"
            path = ROOT / "experiments" / run_id / "manifest.json"
            config = json.loads(path.read_text())["config"]
            configs.append(config)
            sources.append(dict(run_id=run_id, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        config = configs[0]
        if any(c["scene"] != config["scene"] or c["task"]["reference_count"] != config["task"]["reference_count"]
               for c in configs):
            issues.append(task + ": training scene configuration differs across algorithms/sensors")
        hashes, seeds = {}, {}
        for split, count in (("train", config["task"]["reference_count"]), ("dev", 32), ("heldout", 128)):
            scene = instantiate(config["scene"], _convert_="all")
            _, manifest = make_bank(scene, SPLIT_SEEDS[split], count)
            path = raw / (task + "-" + split + ".json")
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            # Exclude seed/ID/difficulty labels: identical physical scenes count
            # as overlap even if metadata labels differ. Obstacle order is irrelevant.
            identities = [fingerprint({
                "corridor": manifest["corridor"], "start": manifest["start"], "goal": manifest["goal"],
                "obstacles": sorted(row["obstacle_table"], key=lambda x: json.dumps(x, sort_keys=True)),
            }) for row in manifest["instances"]]
            hashes[split] = set(identities)
            seeds[split] = {row["instance_seed"] for row in manifest["instances"]}
            if len(hashes[split]) != manifest["instance_count"]:
                issues.append(f"{task}/{split}: duplicate physical scene within split")
            records.append(dict(task=task, split=split, generator_seed=SPLIT_SEEDS[split],
                                instance_count=manifest["instance_count"], bank_digest=manifest["bank_digest"],
                                instance_fingerprints=identities, manifest=str(path.relative_to(ROOT)),
                                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for a, b in itertools.combinations(hashes, 2):
            overlap, seed_overlap = len(hashes[a] & hashes[b]), len(seeds[a] & seeds[b])
            intersections.append(dict(task=task, splits=[a, b], geometry_motion_overlap=overlap,
                                      accepted_seed_overlap=seed_overlap))
            if overlap or seed_overlap:
                issues.append(f"{task}/{a}/{b}: scene or seed overlap")
    result = dict(passed=not issues, issues=issues, banks=records, intersections=intersections,
                  training_sources=sources,
                  identity_scope="Physical obstacle geometry and motion parameters at the manifest's 1e-6 precision; seed/ID labels excluded",
                  construction="Post-run deterministic reconstruction from frozen training configurations and unchanged scene generator; not claimed as a pre-training materialized manifest",
                  generator_sha256=hashlib.sha256((ROOT / "src/drone_playground/tasks/scenes/navigation.py").read_bytes()).hexdigest())
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "scene-splits.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target / "scene-splits.json")
    print(json.dumps(dict(passed=result["passed"], intersections=intersections, issues=issues)))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
