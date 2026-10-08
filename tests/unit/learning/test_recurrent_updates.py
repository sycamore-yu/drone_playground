"""Shared recurrent orchestration preserves update and snapshot boundaries."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from drone_playground.learning.algorithms.recurrent_bptt import TrainingState


def test_resumed_stage_updates_once_per_step_and_saves_final_state(tmp_path):
    from drone_playground.learning.checkpointing import run_recurrent_updates

    state = TrainingState({"w": jnp.float32(0)}, {}, jax.random.PRNGKey(0), jnp.int32(2))
    saved, logged = [], []
    recorder = SimpleNamespace(
        path=tmp_path,
        phase=lambda *args, **kwargs: None,
        log=lambda step, values: logged.append(step),
    )

    def update(current):
        return current.replace(
            params={"w": current.params["w"] + 1}, updates=current.updates + 1
        ), {"loss": jnp.float32(1)}

    def snapshot(current):
        saved.append(int(current.updates))
        return str(tmp_path / f"{int(current.updates)}.pkl")

    config = {"training": {"policy_updates": 8, "stop_after_updates": 5, "max_wall_seconds": 100}}
    current, metrics, timing = run_recurrent_updates(
        state,
        update,
        snapshot,
        config,
        recorder,
        steps_per_update=4,
        milestones={4},
    )
    assert int(current.updates) == 5
    assert float(current.params["w"]) == 3
    assert saved == [2, 4, 5]
    assert metrics == {"loss": 1.0}
    assert len(timing["durations"]) == 3
    assert timing["last_snapshot"] == str(tmp_path / "5.pkl")
