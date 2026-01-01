import numpy as np
from django.test import SimpleTestCase

from quant_tick.lib.calibration import apply_calibration, calibrate_probabilities


class CalibrationTest(SimpleTestCase):
    """Calibration tests."""

    def test_calibrate_probabilities_too_few_samples(self):
        """Skips calibration when sample count is too small."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

        calibrator, method = calibrate_probabilities(y_true, y_pred)

        self.assertIsNone(calibrator)
        self.assertEqual(method, "none")

    def test_calibrate_probabilities_low_variance(self):
        """Skips calibration when predictions lack variance."""
        y_true = np.array([0, 1] * 10)
        y_pred = np.array([0.5] * 20)

        calibrator, method = calibrate_probabilities(y_true, y_pred)

        self.assertIsNone(calibrator)
        self.assertEqual(method, "none")

    def test_calibrate_probabilities_class_imbalance(self):
        """Skips calibration when class imbalance is extreme."""
        y_true = np.array([1] * 2 + [0] * 18)
        y_pred = np.linspace(0.1, 0.9, 20)

        calibrator, method = calibrate_probabilities(y_true, y_pred)

        self.assertIsNone(calibrator)
        self.assertEqual(method, "none")

    def test_auto_uses_isotonic_for_large_sample(self):
        """Auto selects isotonic for larger sample sizes."""
        y_pred = np.array([0.05, 0.15, 0.25, 0.75, 0.85, 0.95] * 10)
        y_true = np.array([0, 0, 0, 1, 1, 1] * 10)

        calibrator, method = calibrate_probabilities(y_true, y_pred, method="auto")

        self.assertIsNotNone(calibrator)
        self.assertEqual(method, "isotonic")

    def test_auto_uses_platt_for_small_sample(self):
        """Auto selects platt for smaller sample sizes."""
        y_pred = np.array([0.05, 0.15, 0.25, 0.75, 0.85, 0.95] * 4)
        y_true = np.array([0, 0, 0, 1, 1, 1] * 4)

        calibrator, method = calibrate_probabilities(y_true, y_pred, method="auto")

        self.assertIsNotNone(calibrator)
        self.assertEqual(method, "platt")

    def test_apply_calibration_returns_expected_shape(self):
        """Calibrated probabilities match input shape."""
        y_pred = np.array([0.05, 0.15, 0.25, 0.75, 0.85, 0.95] * 10)
        y_true = np.array([0, 0, 0, 1, 1, 1] * 10)

        calibrator, method = calibrate_probabilities(y_true, y_pred, method="auto")
        calibrated = apply_calibration(y_pred, calibrator, method)

        self.assertEqual(calibrated.shape, y_pred.shape)
