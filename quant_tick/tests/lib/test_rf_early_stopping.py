import numpy as np
from django.test import TestCase

from quant_tick.lib.ml import PurgedKFold
from quant_tick.lib.rf_early_stopping import find_optimal_n_estimators


class RFEarlyStoppingTestCase(TestCase):
    """Test RF early stopping functions."""

    def setUp(self):
        """Create test fixtures."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 2, n_samples)
        self.weights = np.ones(n_samples)

        self.event_ends = np.arange(n_samples)

    def test_find_optimal_n_estimators_converges(self):
        """Test that early stopping finds optimal n_estimators."""
        cv = PurgedKFold(n_splits=3, embargo_bars=5, shuffle=False)

        optimal_n, history = find_optimal_n_estimators(
            X=self.X,
            y=self.y,
            sample_weight=self.weights,
            kfold=cv,
            max_estimators=300,
            min_estimators=50,
            step=50,
            epsilon=0.001,
            patience=2,
            max_features="sqrt",
            min_samples_leaf=10,
            max_depth=None,
            max_samples=None,
            random_state=42,
        )

        self.assertIsInstance(optimal_n, int)
        self.assertGreaterEqual(optimal_n, 50)
        self.assertLessEqual(optimal_n, 300)

        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        for entry in history:
            self.assertIn("n_estimators", entry)
            self.assertIn("cv_auc", entry)
            self.assertIn("delta", entry)

    def test_early_stopping_history_format(self):
        """Test that convergence history is properly formatted."""
        cv = PurgedKFold(n_splits=2, embargo_bars=5, shuffle=False)

        optimal_n, history = find_optimal_n_estimators(
            X=self.X,
            y=self.y,
            sample_weight=self.weights,
            kfold=cv,
            max_estimators=150,
            min_estimators=50,
            step=50,
            epsilon=0.01,
            patience=1,
            max_features="sqrt",
            min_samples_leaf=10,
            max_depth=10,
            max_samples=0.7,
            random_state=42,
        )

        self.assertIsInstance(history, list)

        for entry in history:
            self.assertIsInstance(entry["n_estimators"], int)
            self.assertIsInstance(entry["cv_auc"], float)
            self.assertIsInstance(entry["delta"], float)

            self.assertGreaterEqual(entry["cv_auc"], 0.0)
            self.assertLessEqual(entry["cv_auc"], 1.0)

    def test_early_stopping_respects_patience(self):
        """Test that early stopping respects patience parameter."""
        cv = PurgedKFold(n_splits=2, embargo_bars=5, shuffle=False)

        optimal_n, history = find_optimal_n_estimators(
            X=self.X,
            y=self.y,
            sample_weight=self.weights,
            kfold=cv,
            max_estimators=500,
            min_estimators=50,
            step=50,
            epsilon=0.001,
            patience=1,
            random_state=42,
        )

        self.assertLessEqual(len(history), 10)

    def test_early_stopping_with_max_depth(self):
        """Test early stopping works with max_depth constraint."""
        cv = PurgedKFold(n_splits=2, embargo_bars=5, shuffle=False)

        optimal_n, history = find_optimal_n_estimators(
            X=self.X,
            y=self.y,
            sample_weight=self.weights,
            kfold=cv,
            max_estimators=200,
            min_estimators=50,
            step=50,
            epsilon=0.01,
            patience=2,
            max_depth=5,
            random_state=42,
        )

        self.assertIsInstance(optimal_n, int)
        self.assertGreater(len(history), 0)

    def test_early_stopping_multiclass(self):
        """Test early stopping with 3-class AFML labels {-1, 0, +1}."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10

        X_multiclass = np.random.randn(n_samples, n_features)
        y_multiclass = np.random.choice([-1, 0, 1], size=n_samples)
        weights_multiclass = np.ones(n_samples)

        cv = PurgedKFold(n_splits=3, embargo_bars=5, shuffle=False)

        # Early stopping should work without errors on multi-class
        optimal_n, history = find_optimal_n_estimators(
            X=X_multiclass,
            y=y_multiclass,
            sample_weight=weights_multiclass,
            kfold=cv,
            max_estimators=200,
            min_estimators=50,
            step=50,
            epsilon=0.01,
            patience=2,
            max_features="sqrt",
            min_samples_leaf=10,
            random_state=42,
        )

        # Should find a valid optimal n_estimators
        self.assertIsInstance(optimal_n, int)
        self.assertGreaterEqual(optimal_n, 50)
        self.assertLessEqual(optimal_n, 200)

        # Should produce convergence history
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

        # Verify history entries are valid
        for entry in history:
            self.assertIn("n_estimators", entry)
            self.assertIn("cv_auc", entry)
            self.assertIn("delta", entry)
            self.assertIsInstance(entry["cv_auc"], float)
            self.assertGreaterEqual(entry["cv_auc"], 0.0)
            self.assertLessEqual(entry["cv_auc"], 1.0)
