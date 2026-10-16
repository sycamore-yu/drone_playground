"""Benchmark actual Crazyflow learning updates at different vectorized world counts.

This measures steady-state updates and GPU-synchronized interactions, not
Python enqueue time. It refuses GPU execution while formal training is active.
Outputs belong in the Git-ignored results/performance directory.

Examples:
  python tools/benchmark_training_throughput.py --dry-run
  python tools/benchmark_training_throughput.py --experiment navigation_lidar_scratch \
      --algorithm bptt --worlds 32,64,128,256 --warmup 2 --updates 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import jax
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.factory import create_environment, create_method


def training_pids():
    """Detect genuinely running training commands, not legacy tmux shells."""
    active = []
    for pid_path in Path("/proc").iterdir():
        if not pid_path.name.isdecimal():
            continue
        try:
            args = (pid_path / "cmdline").read_bytes().split(b"\x00")
        except (OSError, PermissionError):
            continue
        args = [arg.decode(errors="replace") for arg in args if arg]
        if "-m" in args and "drone_playground.cli" in args and "mode=train" in args:
            active.append(int(pid_path.name))
    return active


def gpu_telemetry():
    """Read device occupancy without claiming utilization means FLOPs utilization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        memory, utilization, power = result.stdout.splitlines()[0].split(",")
        return {
            "gpu_memory_used_mib": float(memory),
            "gpu_utilization_percent": float(utilization),
            "gpu_power_w": float(power),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def run_one(config_dir, experiment, algorithm, worlds, warmup, updates):
    """Measure actual compiled Trainer.update calls under a fixed task recipe."""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name="config",
            overrides=[
                f"experiment={experiment}",
                f"learning={algorithm}",
                "mode=train",
                "simulation.device=gpu",
                f"simulation.num_envs={worlds}",
                "initial_checkpoint=null",
            ],
        )
    resolved = OmegaConf.to_container(config, resolve=True)
    env = create_environment(config)
    try:
        trainer = Trainer(
            env,
            kind=config.method.actor.get("kind", "custom"),
            algorithm=algorithm,
            seed=0,
            config=resolved["learning"]["options"],
            actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
            method=create_method(config),
            loss=config.learning.loss,
            backward_model=config.learning.backward_model,
            backward_options=OmegaConf.to_container(config.learning.backward_options, resolve=True),
        )
        state = trainer.initialize()
        jax.block_until_ready(state.params)
        for _ in range(warmup):
            state, _ = trainer.update(state)
            jax.block_until_ready((state.params, state.env_state))
        measured = []
        samples_per_update = worlds * int(trainer.config["horizon"])
        for _ in range(updates):
            start = time.perf_counter()
            state, metrics = trainer.update(state)
            jax.block_until_ready((state.params, state.env_state, metrics))
            elapsed = time.perf_counter() - start
            measured.append(
                {
                    "seconds": elapsed,
                    "steps_per_second": samples_per_update / elapsed,
                    **gpu_telemetry(),
                }
            )
        total_seconds = sum(item["seconds"] for item in measured)
        return {
            "experiment": experiment,
            "algorithm": algorithm,
            "worlds": worlds,
            "horizon": trainer.config["horizon"],
            "warmup_updates": warmup,
            "measured_updates": updates,
            "interactions": samples_per_update * updates,
            "steady_seconds": total_seconds,
            "mean_update_seconds": total_seconds / updates,
            "steady_interactions_per_second": samples_per_update * updates / total_seconds,
            "peak_gpu_memory_used_mib": max(
                (item.get("gpu_memory_used_mib", 0) for item in measured), default=0
            ),
            "individual_updates": measured,
        }
    finally:
        env.sim.close()


def main():
    """Run a controlled single-GPU experiment only if the formal matrix is idle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="navigation_lidar_scratch")
    parser.add_argument("--algorithm", choices=("ppo", "bptt", "shac"), default="bptt")
    parser.add_argument("--worlds", default="32,64,128,256")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/performance/scratch_world_count.json"),
    )
    args = parser.parse_args()
    worlds = [int(value) for value in args.worlds.split(",")]
    if (
        not worlds
        or any(value < 1 for value in worlds)
        or len(set(worlds)) != len(worlds)
        or args.warmup < 1
        or args.updates < 1
    ):
        parser.error("worlds must be unique positive integers; warmup/updates must be positive")
    spec = {
        "experiment": args.experiment,
        "algorithm": args.algorithm,
        "worlds": worlds,
        "warmup": args.warmup,
        "updates": args.updates,
        "measurement": "jax.block_until_ready after each actual Trainer.update",
    }
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return
    active = training_pids()
    if active:
        parser.error(
            f"Refusing to compete with {len(active)} active GPU training processes: {active}"
        )
    config_dir = (Path(__file__).resolve().parents[1] / "src/drone_playground/configs").resolve()
    if args.worker:
        if len(worlds) != 1:
            parser.error("Worker must receive exactly one world count")
        if not jax.devices("gpu"):
            parser.error("No GPU available")
        print(
            json.dumps(
                run_one(
                    config_dir,
                    args.experiment,
                    args.algorithm,
                    worlds[0],
                    args.warmup,
                    args.updates,
                )
            )
        )
        return
    summary = {
        "source": "actual_Crazyflow",
        "timestamp": datetime.now(UTC).isoformat(),
        **spec,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for count in worlds:
        # Each batch size gets its own JAX GPU process so allocator pools and
        # compiled variants from a prior batch cannot bias memory comparisons.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--experiment",
            args.experiment,
            "--algorithm",
            args.algorithm,
            "--worlds",
            str(count),
            "--warmup",
            str(args.warmup),
            "--updates",
            str(args.updates),
        ]
        sample = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if sample.returncode:
            summary["errors"] = [
                *summary.get("errors", []),
                {"worlds": count, "error": sample.stderr[-3000:]},
            ]
            args.output.write_text(json.dumps(summary, indent=2))
            raise RuntimeError(
                f"Throughput experiment failed at worlds={count}: {sample.stderr[-1200:]}"
            )
        outcome = json.loads(sample.stdout.splitlines()[-1])
        summary["results"].append(outcome)
        args.output.write_text(json.dumps(summary, indent=2))
        print(
            f"worlds={count}, steady_sps={outcome['steady_interactions_per_second']:.1f}, "
            f"update_ms={1000 * outcome['mean_update_seconds']:.1f}, "
            f"gpu_peak_mib={outcome['peak_gpu_memory_used_mib']:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
