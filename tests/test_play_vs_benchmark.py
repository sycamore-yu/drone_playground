"""Keep interactive flight inspection separate from benchmark-only measurements."""

from pathlib import Path

from hydra import compose, initialize_config_dir

from drone_playground.benchmark.evaluation import evaluate


def test_play_skips_benchmark_only_profiling_and_environment_metrics(monkeypatch, tmp_path):
    """Play should finish a flight without latency and environment-complexity sweeps."""
    from drone_playground.benchmark import evaluation

    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "src/drone_playground/configs"),
    ):
        config = compose(
            config_name="config",
            overrides=[
                "mode=play",
                "task=tracking",
                "scene=empty",
                "sensor=state",
                "method=controller",
                "controller=mellinger",
                "simulation.device=cpu",
                "simulation.num_envs=1",
                "task.duration=0.04",
            ],
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Play must not run benchmark profiling")

    monkeypatch.setattr(evaluation, "profile", forbidden)
    monkeypatch.setattr(evaluation, "traversability", forbidden)
    report = evaluate(config, tmp_path)
    assert report["complete"]
    assert report["reports"]["empty"]["episodes"] == 1
    assert report["reports"]["empty"]["timing"] is None
    assert report["reports"]["empty"]["traversability"] is None
    assert (tmp_path / "episodes.csv").is_file()
