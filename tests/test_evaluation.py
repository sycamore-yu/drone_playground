"""Independent evaluation must count every trial and stop scoring after failure."""

import unittest

import numpy as np


class EvaluationContractTests(unittest.TestCase):
    def test_failure_is_retained_and_future_frames_are_not_counted(self):
        from drone_playground.evaluation.tracking import summarize_trials

        trace = {
            "active": np.array([[1, 1], [1, 1], [0, 1], [0, 1]], bool),
            "failed": np.array([[0, 0], [1, 0], [1, 0], [1, 0]], bool),
            "reward": np.array([[1.0, 1.0], [-1.0, 1.0], [999.0, 1.0], [999.0, 1.0]]),
            "metrics": {"tracking_error": np.array([[0.2, 0.1], [0.4, 0.1], [99, 0.1], [99, 0.1]])},
        }
        report = summarize_trials(trace, seeds=[10, 11], dt=0.02)
        self.assertEqual(report["num_trials"], 2)
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["episodes"][0]["steps"], 2)
        self.assertAlmostEqual(report["episodes"][0]["return"], 0.0)
        self.assertAlmostEqual(report["rmse_completed_mean"], 0.1)
        self.assertFalse(report["quality_passed"])

    def test_zero_completed_trials_is_an_explicit_quality_failure(self):
        from drone_playground.evaluation.tracking import summarize_trials

        report = summarize_trials(
            {
                "active": np.ones((1, 2), bool),
                "failed": np.ones((1, 2), bool),
                "reward": -np.ones((1, 2)),
                "metrics": {"tracking_error": np.ones((1, 2))},
            },
            seeds=[0, 1],
            dt=0.02,
        )
        self.assertIsNone(report["rmse_completed_mean"])
        self.assertFalse(report["quality_passed"])


if __name__ == "__main__":
    unittest.main()
