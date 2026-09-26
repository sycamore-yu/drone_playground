"""A frozen policy may explicitly choose a different compatible execution model."""

import unittest

from drone_playground.composition import compose_config
from drone_playground.evaluation import execution


class ExecutionConfigTests(unittest.TestCase):
    def test_checkpoint_environment_is_explicit_default(self):
        self.assertTrue(hasattr(execution, "resolve_evaluation_config"))
        requested = compose_config("figure8_ppo")
        requested["mode"] = "evaluate"
        saved = compose_config("lotf_hybrid_hover")
        result = execution.resolve_evaluation_config(requested, {"config": saved})
        self.assertEqual(result["dynamics"]["forward"], "lotf_high_fidelity")

    def test_explicit_experiment_environment_keeps_frozen_network(self):
        self.assertTrue(hasattr(execution, "resolve_evaluation_config"))
        requested = compose_config("figure8_ppo", ["dynamics.forward=first_principles"])
        requested["mode"] = "evaluate"
        requested["evaluation"]["environment"] = "experiment"
        saved = compose_config("figure8_apg")
        result = execution.resolve_evaluation_config(requested, {"config": saved})
        self.assertEqual(result["dynamics"]["forward"], "first_principles")
        self.assertEqual(result["network"], saved["network"])
        self.assertEqual(result["algorithm"], saved["algorithm"])
