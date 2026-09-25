"""The native watcher must see an atomically complete additional rollout."""

import time
from pathlib import Path

import numpy as np


def test_incremental_publication_keeps_watched_directory_and_native_created_event(tmp_path: Path):
    from crazyflow.envs import FigureEightEnv
    from rscope import rollout
    from rscope.event_handler import MjUnrollHandler
    from watchdog.observers import Observer

    from drone_playground.runs.rscope_io import append_rollout, export_rollout, publish_run

    env = FigureEightEnv(num_envs=1, freq=50, device="cpu")
    env.sim.reset()
    observer = Observer()
    try:
        trace = {
            "pos": np.zeros((3, 1, 3)),
            "quat": np.tile([0, 0, 0, 1], (3, 1, 1)),
            "time": np.arange(1, 4)[:, None] / 50,
            "obs": np.ones((3, 1, 43)),
            "reward": np.ones((3, 1)),
            "metrics": {"training_step": np.ones((3, 1))},
        }
        export_rollout(env.sim, tmp_path / "first", trace)
        export_rollout(env.sim, tmp_path / "second", trace)
        active = publish_run(tmp_path / "first", tmp_path / "active")
        inode = active.stat().st_ino
        before = len(rollout.rollouts)
        observer.schedule(MjUnrollHandler(), str(active), recursive=False)
        observer.start()
        append_rollout(tmp_path / "second", active)
        deadline = time.monotonic() + 3
        while len(rollout.rollouts) == before and time.monotonic() < deadline:
            time.sleep(0.02)
        assert active.stat().st_ino == inode
        assert len(list(active.glob("*.mj_unroll"))) == 2
        assert len(rollout.rollouts) == before + 1
        np.testing.assert_array_equal(rollout.rollouts[-1].reward, trace["reward"])
    finally:
        observer.stop()
        if observer.is_alive():
            observer.join()
        env.close()
