import numpy as np
from django.test import TestCase
from sklearn.ensemble import RandomForestClassifier

from quant_tick.lib.rf_diagnostics import (
    compute_oob_metrics,
    compute_permutation_importances,
    generate_diagnostics_report,
    iterative_feature_pruning,
)


class RFDiagnosticsTestCase(TestCase):
    """Test RF diagnostics functions."""

    def setUp(self):
        """Create test fixtures."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 2, n_samples)
        self.feature_names = [f"feature_{i}" for i in range(n_features)]

    def test_compute_permutation_importances(self):
        """Test permutation importance computation."""
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)

        importances = compute_permutation_importances(
            model, self.X, self.y, n_repeats=5, random_state=42
        )

        self.assertEqual(len(importances), self.X.shape[1])
        self.assertTrue(all(isinstance(v, float) for v in importances.values()))
        self.assertTrue(all(v >= 0 for v in importances.values()))

    def test_compute_oob_metrics(self):
        """Test OOB metrics computation."""
        oob_metrics = compute_oob_metrics(
            self.X,
            self.y,
            n_estimators=50,
            max_samples=0.7,
            random_state=42,
        )

        self.assertIn("oob_score", oob_metrics)
        self.assertIn("train_score", oob_metrics)
        self.assertIn("train_val_gap", oob_metrics)

        self.assertGreater(oob_metrics["oob_score"], 0.0)
        self.assertGreater(oob_metrics["train_score"], 0.0)
        self.assertIsInstance(oob_metrics["train_val_gap"], float)

    def test_iterative_feature_pruning(self):
        """Test iterative feature pruning."""
        best_features, pruning_history = iterative_feature_pruning(
            self.X,
            self.y,
            self.feature_names,
            n_estimators=50,
            max_samples=0.7,
            n_splits=3,
            min_features=3,
            prune_fraction=0.3,
            random_state=42,
        )

        self.assertIsInstance(best_features, list)
        self.assertGreater(len(best_features), 0)
        self.assertLessEqual(len(best_features), len(self.feature_names))

        self.assertIsInstance(pruning_history, list)
        self.assertGreater(len(pruning_history), 0)

        for iteration in pruning_history:
            self.assertIn("iteration", iteration)
            self.assertIn("n_features", iteration)
            self.assertIn("cv_auc", iteration)
            self.assertIn("features", iteration)
            self.assertIn("oob_score", iteration)

    def test_generate_diagnostics_report(self):
        """Test diagnostics report generation."""
        perm_importances = {i: float(i * 0.1) for i in range(10)}

        oob_metrics = {
            "oob_score": 0.75,
            "train_score": 0.82,
            "train_val_gap": 0.07,
        }

        pruning_history = [
            {
                "iteration": 0,
                "n_features": 10,
                "cv_auc": 0.72,
                "cv_std": 0.05,
                "features": self.feature_names,
            },
            {
                "iteration": 1,
                "n_features": 7,
                "cv_auc": 0.75,
                "cv_std": 0.04,
                "features": self.feature_names[:7],
            },
        ]

        report = generate_diagnostics_report(
            perm_importances,
            self.feature_names,
            pruning_history,
            oob_metrics,
        )

        self.assertIn("permutation_importances", report)
        self.assertIn("top_20", report["permutation_importances"])
        self.assertIn("bottom_20", report["permutation_importances"])

        self.assertIn("pruning", report)
        self.assertIn("iterations", report["pruning"])
        self.assertIn("best_iteration", report["pruning"])

        self.assertIn("oob_validation", report)
        self.assertEqual(report["oob_validation"]["oob_score"], 0.75)

    def test_iterative_pruning_stops_at_min_features(self):
        """Test that pruning stops at minimum features."""
        best_features, pruning_history = iterative_feature_pruning(
            self.X,
            self.y,
            self.feature_names,
            n_estimators=50,
            n_splits=3,
            min_features=8,
            prune_fraction=0.5,
            random_state=42,
        )

        self.assertGreaterEqual(len(best_features), 8)

    def test_oob_metrics_with_high_max_samples(self):
        """Test OOB metrics with high max_samples value."""
        oob_metrics = compute_oob_metrics(
            self.X,
            self.y,
            n_estimators=50,
            max_samples=0.9,
            random_state=42,
        )

        self.assertLess(oob_metrics["train_val_gap"], 0.2)
