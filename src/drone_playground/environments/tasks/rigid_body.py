"""Shared access to task-local containers carrying native rigid-body data."""

import numpy as np


class RigidBodyTask:
    """Task hooks shared by tracking, racing and rigid-body navigation."""

    @staticmethod
    def physics(data):
        """Read the native physics state without copying task or sensor memory."""
        return data.sim_data

    @staticmethod
    def with_physics(data, physical):
        """Preserve task-local state when the simulator advances."""
        return data.replace(sim_data=physical)

    def probe(self, env, state):
        """Tasks with substep collision evidence override this hook."""
        del env, state
        return None

    def controller_observation(self, env, state):
        """Expose only the physical fields needed by host controllers."""
        del env
        data = self.physics(state.pipeline_state).states
        return {
            name: np.asarray(getattr(data, name)[0, 0])
            for name in ("pos", "quat", "vel", "ang_vel")
        }

    @property
    def freq(self):
        """Read the environment-owned control frequency."""
        return round(1.0 / self.transition.dt)

    @property
    def physics_freq(self):
        """Read the environment-owned physical frequency."""
        return round(1.0 / self.transition.physics_dt)

    @property
    def dt(self):
        return self.transition.dt

    @property
    def physics_dt(self):
        return self.transition.physics_dt

    @property
    def substeps(self):
        return self.transition.substeps

    @property
    def episode_length(self):
        return round(self.duration * self.freq)
