"""Parameter-loader migration from LSY's old Crazyflow dependency to 0.3.2."""

from crazyflow.dynamics import load_params


def load_hardware_params(drone):
    return load_params("first_principles", drone)
