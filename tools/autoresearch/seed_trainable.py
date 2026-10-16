"""Mutable optimization settings; task, budget and evaluation belong to the benchmark."""


def get_candidate():
    """Start with the native procedural Depth PPO validation recipe."""
    return {
        "description": "Native procedural Depth PPO, from scratch, fixed interaction budget.",
        "recipe": {
            "learning.options.lr": 0.0001,
            "learning.options.horizon": 32,
            "simulation.num_envs": 16,
        },
    }


def train_agent(*args, **kwargs):
    """The upstream loader requires this name; the external backend owns dispatch."""
    raise RuntimeError("Use the DronePlaygroundBackend external benchmark")


RewardRecipeWrapper = train_agent
save_agent_checkpoint = train_agent
