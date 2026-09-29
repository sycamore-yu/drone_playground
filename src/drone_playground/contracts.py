"""Command contracts are independent of algorithm names and native state layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    fields: tuple[str, ...]
    units: tuple[str, ...]
    frame: str


ATTITUDE_THRUST = CommandSpec(
    "attitude_thrust",
    ("roll", "pitch", "yaw", "thrust"),
    ("rad", "rad", "rad", "N"),
    "world attitude / body thrust",
)
THRUST_BODYRATES = CommandSpec(
    "thrust_bodyrates",
    ("thrust", "roll_rate", "pitch_rate", "yaw_rate"),
    ("N", "rad/s", "rad/s", "rad/s"),
    "body",
)
MOTOR_RPM = CommandSpec(
    "motor_rpm", ("motor0", "motor1", "motor2", "motor3"), ("rpm",) * 4, "rotor"
)
TRAJECTORY = CommandSpec("trajectory", ("position", "velocity", "time"), ("m", "m/s", "s"), "world")
WORLD_ACCELERATION = CommandSpec(
    "world_acceleration",
    ("ax", "ay", "az"),
    ("m/s^2",) * 3,
    "world; gravity-compensated net acceleration",
)
VELOCITY_YAW = CommandSpec(
    "velocity_yaw", ("vx", "vy", "vz", "yaw"), ("m/s", "m/s", "m/s", "rad"), "world"
)
COMMANDS = {
    spec.name: spec
    for spec in (
        ATTITUDE_THRUST,
        THRUST_BODYRATES,
        MOTOR_RPM,
        TRAJECTORY,
        WORLD_ACCELERATION,
        VELOCITY_YAW,
    )
}


def require_match(output: str, input_: str) -> None:
    if output not in COMMANDS or input_ not in COMMANDS:
        raise ValueError(f"Unknown command contract: {output} -> {input_}")
    if output != input_:
        raise ValueError(f"Command interface mismatch: {output} -> {input_}")
