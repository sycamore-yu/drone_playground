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

    def close(self, env):
        """Release the native simulation resource owned by this environment."""
        env.sim.close()
