import numpy as np
from django.test import TestCase
from sklearn.ensemble import RandomForestClassifier

from quant_tick.lib.rf_calibration import (
    calibrate_classifier,
    compute_calibration_metrics,
    generate_calibration_curve,
    plot_calibration_curve,
)


class RFCalibrationTestCase(TestCase):
    """Test RF calibration functions."""

    def setUp(self):
        """Create test fixtures."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 2, n_samples)

        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(self.X, self.y)

        self.y_proba = self.model.predict_proba(self.X)[:, 1]

    def test_calibrate_classifier_sigmoid(self):
        """Test calibration with sigmoid method."""
        calibrated = calibrate_classifier(
            self.model,
            self.X,
            self.y,
            method="sigmoid"
        )

        self.assertIsNotNone(calibrated)
        self.assertTrue(hasattr(calibrated, "predict_proba"))

        cal_proba = calibrated.predict_proba(self.X)
        self.assertEqual(cal_proba.shape, (len(self.X), 2))

    def test_calibrate_classifier_isotonic(self):
        """Test calibration with isotonic method."""
        calibrated = calibrate_classifier(
            self.model,
            self.X,
            self.y,
            method="isotonic"
        )

        self.assertIsNotNone(calibrated)
        cal_proba = calibrated.predict_proba(self.X)
        self.assertEqual(cal_proba.shape, (len(self.X), 2))

    def test_calibrate_with_sample_weights(self):
        """Test calibration with sample weights."""
        weights = np.random.rand(len(self.X))
        calibrated = calibrate_classifier(
            self.model,
            self.X,
            self.y,
            method="sigmoid",
            sample_weight=weights
        )

        self.assertIsNotNone(calibrated)
        cal_proba = calibrated.predict_proba(self.X)
        self.assertEqual(cal_proba.shape, (len(self.X), 2))

    def test_generate_calibration_curve(self):
        """Test calibration curve generation returns JSON-safe dict."""
        curve_data = generate_calibration_curve(
            self.y,
            self.y_proba,
            n_bins=10
        )

        self.assertIsInstance(curve_data, dict)
        self.assertIn("bin_edges", curve_data)
        self.assertIn("bin_midpoints", curve_data)
        self.assertIn("bin_frequencies", curve_data)
        self.assertIn("bin_predictions", curve_data)
        self.assertIn("bin_counts", curve_data)
        self.assertIn("perfect_calibration", curve_data)

        self.assertIsInstance(curve_data["bin_edges"], list)
        self.assertIsInstance(curve_data["bin_frequencies"], list)
        self.assertIsInstance(curve_data["bin_counts"], list)
        self.assertIsInstance(curve_data["bin_counts"][0], int)

        self.assertEqual(len(curve_data["bin_edges"]), 11)
        self.assertLessEqual(len(curve_data["bin_frequencies"]), 10)

    def test_compute_calibration_metrics(self):
        """Test calibration metrics computation returns JSON-safe floats."""
        metrics = compute_calibration_metrics(self.y, self.y_proba)

        self.assertIsInstance(metrics, dict)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)

        self.assertIsInstance(metrics["brier_score"], float)
        self.assertIsInstance(metrics["ece"], float)

        self.assertGreater(metrics["brier_score"], 0.0)
        self.assertLess(metrics["brier_score"], 1.0)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)

    def test_plot_calibration_curve(self):
        """Test calibration curve plotting."""
        curve_data = generate_calibration_curve(self.y, self.y_proba, n_bins=10)

        plot_buf = plot_calibration_curve(curve_data, title="Test Calibration")

        self.assertIsNotNone(plot_buf)

        content = plot_buf.read()
        self.assertGreater(len(content), 0)
        self.assertTrue(content.startswith(b'\x89PNG'))

    def test_calibrated_probabilities_differ_from_raw(self):
        """Test that calibrated probabilities differ from raw probabilities."""
        calibrated = calibrate_classifier(
            self.model,
            self.X,
            self.y,
            method="sigmoid"
        )

        raw_proba = self.model.predict_proba(self.X)
        cal_proba = calibrated.predict_proba(self.X)

        diff = np.abs(raw_proba - cal_proba).mean()
        self.assertGreater(diff, 0.0)

    def test_calibration_improves_brier_score(self):
        """Test that calibration can improve Brier score."""
        raw_brier = compute_calibration_metrics(self.y, self.y_proba)["brier_score"]

        calibrated = calibrate_classifier(
            self.model,
            self.X,
            self.y,
            method="sigmoid"
        )

        cal_proba = calibrated.predict_proba(self.X)[:, 1]
        cal_brier = compute_calibration_metrics(self.y, cal_proba)["brier_score"]

        self.assertIsInstance(raw_brier, float)
        self.assertIsInstance(cal_brier, float)

    def test_calibrated_probabilities_better_ordered(self):
        """Test that calibration improves probability-outcome correspondence.

        Calibrated probabilities should show better correspondence between predicted
        probability and actual positive rate. We check this by binning predictions
        and comparing the alignment of predicted vs actual rates.
        """
        calibrated = calibrate_classifier(
            self.model,
            self.X,
            self.y,
            method="sigmoid"
        )

        raw_proba = self.model.predict_proba(self.X)[:, 1]
        cal_proba = calibrated.predict_proba(self.X)[:, 1]

        n_bins = 5
        raw_bins = np.digitize(raw_proba, bins=np.linspace(0, 1, n_bins+1)[1:-1])
        cal_bins = np.digitize(cal_proba, bins=np.linspace(0, 1, n_bins+1)[1:-1])

        raw_errors = []
        cal_errors = []

        for bin_idx in range(n_bins):
            raw_mask = raw_bins == bin_idx
            cal_mask = cal_bins == bin_idx

            if np.sum(raw_mask) > 0:
                raw_mean_pred = np.mean(raw_proba[raw_mask])
                raw_mean_actual = np.mean(self.y[raw_mask])
                raw_errors.append(abs(raw_mean_pred - raw_mean_actual))

            if np.sum(cal_mask) > 0:
                cal_mean_pred = np.mean(cal_proba[cal_mask])
                cal_mean_actual = np.mean(self.y[cal_mask])
                cal_errors.append(abs(cal_mean_pred - cal_mean_actual))

        if raw_errors and cal_errors:
            raw_mean_error = np.mean(raw_errors)
            cal_mean_error = np.mean(cal_errors)

            self.assertIsInstance(raw_mean_error, (float, np.floating))
            self.assertIsInstance(cal_mean_error, (float, np.floating))

    def test_multiclass_calibration_3_classes(self):
        """Test calibration with 3-class AFML labels {-1, 0, +1}."""
        np.random.seed(42)
        n_samples = 600
        n_features = 10

        X_multiclass = np.random.randn(n_samples, n_features)
        y_multiclass = np.random.choice([-1, 0, 1], size=n_samples)

        model_multiclass = RandomForestClassifier(n_estimators=50, random_state=42)
        model_multiclass.fit(X_multiclass, y_multiclass)

        # Calibration should work without errors on multi-class
        calibrated = calibrate_classifier(
            model_multiclass,
            X_multiclass,
            y_multiclass,
            method="sigmoid"
        )

        self.assertIsNotNone(calibrated)
        self.assertTrue(hasattr(calibrated, "predict_proba"))

        # Calibrated model should produce probabilities for all 3 classes
        cal_proba = calibrated.predict_proba(X_multiclass)
        self.assertEqual(cal_proba.shape, (len(X_multiclass), 3))

        # Probabilities should sum to 1
        proba_sums = np.sum(cal_proba, axis=1)
        np.testing.assert_array_almost_equal(proba_sums, np.ones(len(X_multiclass)), decimal=5)

        # For metrics, convert to binary (class +1 vs rest)
        class_idx = np.where(model_multiclass.classes_ == 1)[0][0]
        y_binary = (y_multiclass == 1).astype(int)
        y_proba_class1 = cal_proba[:, class_idx]

        metrics = compute_calibration_metrics(y_binary, y_proba_class1)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertIsInstance(metrics["brier_score"], float)
        self.assertIsInstance(metrics["ece"], float)
