"""Convenience CLI for the same Hydra-driven optimization execution entry."""

import argparse
from pathlib import Path

from drone_playground.composition import compose_config, run_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["attitude_mpc", "sampling_mpc"], required=True)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed-start", type=int)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--prediction-device", choices=["cpu", "gpu"], default="cpu")
    args = p.parse_args()
    cfg = compose_config("racing_" + args.controller)
    cfg["evaluation"].update(episodes=args.episodes, split=args.split, seed_start=args.seed_start)
    if args.controller == "sampling_mpc":
        cfg["controller"].update(samples=args.samples, prediction_device=args.prediction_device)
    return run_experiment(cfg, Path(__file__).resolve().parents[1], args.run_id)


if __name__ == "__main__":
    main()
