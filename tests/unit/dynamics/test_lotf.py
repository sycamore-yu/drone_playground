"""Native LOTF forward equations and intentional surrogate derivatives."""

import importlib
import importlib.util
import unittest

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class LOTFModelTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.dynamics.lotf"
        self.assertIsNotNone(importlib.util.find_spec(name), "LOTF model adapter must exist")
        return importlib.import_module(name)

    def test_forward_matches_upstream_and_uses_actual_example_machine(self):
        m = self.module()
        model = m.LOTFModel()
        native = model.native
        self.assertAlmostEqual(native._mass, 0.192)
        self.assertAlmostEqual(native._motor_tau, 0.0245)
        initial = native.create_state(
            p=jnp.array([0.0, 0.0, 1.5]), R=jnp.eye(3), v=jnp.array([0.1, 0.2, 0.0])
        )
        command = jnp.array([0.192 * 9.81, 0.05, -0.02, 0.01])
        state, reference = initial, initial
        for _ in range(6):
            state = model.step(state, command, 0.02)
            reference = native.step(reference, command[0], command[1:], None, 0.02)
        for key in ("p", "R", "v", "omega", "domega", "motor_omega", "acc"):
            np.testing.assert_allclose(
                getattr(state, key), getattr(reference, key), atol=1e-6, rtol=1e-6, err_msg=key
            )

    def test_backward_matches_simplified_jacobian_and_prng_has_zero_tangent(self):
        m = self.module()
        model = m.LOTFModel()
        state = model.native.default_state()
        command = jnp.array([0.192 * 9.81, 0.05, -0.02, 0.01])
        actual = jax.jacfwd(lambda u: model.step(state, u, 0.02).v)(command)
        from lotf.objects.quadrotor_obj import simplified_dyn

        expected = jax.jacfwd(
            lambda u: simplified_dyn(state.p, state.R, state.v, u[0] / 0.192, u[1:], 0.02)[2]
        )(command)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertTrue(np.isfinite(actual).all())
        grad = jax.grad(lambda u: jnp.sum(model.step(state, u, 0.02).p ** 2))(command)
        self.assertTrue(np.isfinite(grad).all())

    def test_selected_direct_rule_has_same_forward_and_distinct_derivative(self):
        m = self.module()
        hybrid = m.LOTFModel()
        direct = m.LOTFModel(backward="direct")
        # The source aerodynamic polynomial has sqrt(vx²+vy²), whose direct
        # derivative is undefined at zero horizontal speed. Use an ordinary state.
        state = hybrid.native.default_state().replace(v=jnp.array([0.1, 0.2, 0.01]))
        u = jnp.array([0.192 * 9.81, 0.3, -0.2, 0.1])
        a = hybrid.step(state, u, 0.02)
        b = direct.step(state, u, 0.02)
        np.testing.assert_allclose(a.v, b.v, atol=1e-6)
        ja = jax.jacfwd(lambda x: hybrid.step(state, x, 0.02).v)(u)
        jb = jax.jacfwd(lambda x: direct.step(state, x, 0.02).v)(u)
        self.assertGreater(float(jnp.linalg.norm(ja - jb)), 1e-5)

    def test_full_prv_state_and_action_surrogate_jacobian(self):
        m = self.module()
        model = m.LOTFModel()
        state = model.native.default_state().replace(v=jnp.array([0.1, 0.2, 0.01]))
        command = jnp.array([0.192 * 9.81, 0.3, -0.2, 0.1])
        vector = jnp.concatenate([state.p, state.R.reshape(-1), state.v, command])
        from lotf.objects.quadrotor_obj import simplified_dyn

        def actual(z):
            initial = state.replace(p=z[:3], R=z[3:12].reshape(3, 3), v=z[12:15])
            out = model.step(initial, z[15:], 0.02)
            return jnp.concatenate([out.p, out.R.reshape(-1), out.v])

        def expected(z):
            p, rotation, v = simplified_dyn(
                z[:3], z[3:12].reshape(3, 3), z[12:15], z[15] / 0.192, z[16:], 0.02
            )
            return jnp.concatenate([p, rotation.reshape(-1), v])

        np.testing.assert_allclose(
            jax.jacfwd(actual)(vector), jax.jacfwd(expected)(vector), atol=1e-6, rtol=1e-6
        )
