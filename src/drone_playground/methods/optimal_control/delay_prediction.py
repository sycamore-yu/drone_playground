"""Causal state prediction from a model and the controller's issued commands."""

from collections import deque

import numpy as np


class IssuedCommandPredictor:
    """Predict until a new command would arrive, using a fixed delay estimate.

    The model state is [position, XYZ Euler angles, velocity, Euler rates].
    No simulator queue, sampled delay or future observation is accessible here.
    """

    def __init__(self, rhs, delay_seconds, initial_action, *, max_step_seconds=.002):
        if not np.isfinite(delay_seconds) or delay_seconds <= 0:
            raise ValueError("Prediction delay must be finite and positive")
        if not np.isfinite(max_step_seconds) or max_step_seconds <= 0:
            raise ValueError("Prediction integration step must be finite and positive")
        self.rhs = rhs
        self.delay_seconds = float(delay_seconds)
        self.max_step_seconds = float(max_step_seconds)
        self.initial_action = np.asarray(initial_action, dtype=float).copy()
        self.history = deque()

    def reset(self):
        self.history.clear()

    def record(self, now, action):
        if not np.isfinite(now) or (self.history and now <= self.history[-1][0]):
            raise ValueError("Issued command times must be finite and strictly increasing")
        self.history.append((float(now), np.asarray(action, dtype=float).copy()))
        # Retain the last command preceding the next prediction interval too.
        while len(self.history) > 1 and self.history[1][0] < now - self.delay_seconds:
            self.history.popleft()

    def predict(self, observation, now):
        from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
        from scipy.spatial.transform import Rotation

        if not np.isfinite(now) or (self.history and now <= self.history[-1][0]):
            raise ValueError("Prediction must precede recording the current command")
        rpy = Rotation.from_quat(observation['quat']).as_euler('xyz')
        rates = np.asarray(ang_vel2rpy_rates(observation['quat'], observation['ang_vel']))
        x = np.r_[observation['pos'], rpy, observation['vel'], rates]
        tau = 0.
        while tau < self.delay_seconds - 1e-10:
            issued = now + tau - self.delay_seconds
            action = self.initial_action
            for stamp, previous in self.history:
                if stamp <= issued + 1e-9:
                    action = previous
            h = min(self.max_step_seconds, self.delay_seconds - tau)

            def f(value):
                return np.asarray(self.rhs(value, action)).ravel()

            k1 = f(x)
            k2 = f(x + h * k1 / 2)
            k3 = f(x + h * k2 / 2)
            k4 = f(x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            tau += h
        roll, pitch, _ = x[3:6]
        rd, pd, yd = x[9:12]
        omega = np.array([
            rd - yd * np.sin(pitch),
            pd * np.cos(roll) + yd * np.sin(roll) * np.cos(pitch),
            -pd * np.sin(roll) + yd * np.cos(roll) * np.cos(pitch),
        ])
        return {**observation, 'pos': x[:3], 'vel': x[6:9],
                'quat': Rotation.from_euler('xyz', x[3:6]).as_quat(), 'ang_vel': omega}
