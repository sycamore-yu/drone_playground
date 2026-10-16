"""Mutable optimization settings; task, budget and evaluation belong to the benchmark."""


def get_candidate():
    """Start with the task's declared training recipe."""
    return {
        "description": "Declared baseline, trained from scratch for the fixed interaction budget.",
        "recipe": {},
    }


def train_agent(*args, **kwargs):
    """The upstream loader requires this name; the external backend owns dispatch."""
    raise RuntimeError("Use the DronePlaygroundBackend external benchmark")


RewardRecipeWrapper = train_agent
save_agent_checkpoint = train_agent
