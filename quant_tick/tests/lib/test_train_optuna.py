import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.train import (
    _cv_brier_for_params,
    _tune_lgbm_hyperparams_optuna,
)


class CVBrierForParamsTest(TestCase):
    """Test CV Brier score helper."""

    def test_returns_float_brier_score(self):
        """CV Brier score is a float."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        timestamp_idx = np.arange(n_samples)

        lgbm_params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.75,
            "random_state": 42,
            "verbose": -1,
        }

        brier = _cv_brier_for_params(
            X_train_full=X_train,
            y_train_full=y_train,
            timestamp_idx_train=timestamp_idx,
            horizon=10,
            n_splits=3,
            embargo_bars=5,
            model_params=lgbm_params,
        )

        self.assertIsInstance(brier, float)
        self.assertGreater(brier, 0.0)
        self.assertLess(brier, 1.0)

    def test_raises_on_zero_folds(self):
        """Raises ValueError if PurgedKFold yields zero folds."""
        np.random.seed(42)
        n_samples = 20  # Too small for aggressive purging
        n_features = 3

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        timestamp_idx = np.arange(n_samples)

        lgbm_params = {
            "objective": "binary",
            "n_estimators": 10,
            "random_state": 42,
            "verbose": -1,
        }

        # Very aggressive purging/embargo should yield zero folds
        with self.assertRaises(ValueError) as ctx:
            _cv_brier_for_params(
                X_train_full=X_train,
                y_train_full=y_train,
                timestamp_idx_train=timestamp_idx,
                horizon=100,  # Very long horizon
                n_splits=5,
                embargo_bars=50,  # Very long embargo
                model_params=lgbm_params,
            )

        self.assertIn("zero folds", str(ctx.exception))

    def test_different_params_produce_different_scores(self):
        """Different LightGBM params should produce different Brier scores."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        timestamp_idx = np.arange(n_samples)

        base_params = {
            "objective": "binary",
            "subsample": 0.75,
            "random_state": 42,
            "verbose": -1,
        }

        # Two different parameter configurations
        params_1 = {**base_params, "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        params_2 = {**base_params, "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}

        brier_1 = _cv_brier_for_params(
            X_train, y_train, timestamp_idx, 10, 3, 5, params_1
        )
        brier_2 = _cv_brier_for_params(
            X_train, y_train, timestamp_idx, 10, 3, 5, params_2
        )

        # Scores should be different (different params)
        self.assertNotEqual(brier_1, brier_2)


class TuneLGBMHyperparamsOptunaTest(TestCase):
    """Test Optuna hyperparameter tuner."""

    def test_returns_tuned_params(self):
        """Optuna tuner returns params dict with tuned values."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        timestamp_idx = np.arange(n_samples)

        base_params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "subsample": 0.75,
            "random_state": 42,
            "verbose": -1,
        }

        # Run with minimal trials for speed
        tuned_params = _tune_lgbm_hyperparams_optuna(
            X_train_full=X_train,
            y_train_full=y_train,
            timestamp_idx_train=timestamp_idx,
            horizon=10,
            side="lower",
            n_splits=3,
            embargo_bars=5,
            base_lgbm_params=base_params,
            n_trials=2,
        )

        # Should return a dict
        self.assertIsInstance(tuned_params, dict)

        # Should contain tuned hyperparameters
        self.assertIn("learning_rate", tuned_params)
        self.assertIn("max_depth", tuned_params)
        self.assertIn("n_estimators", tuned_params)

        # Should preserve base params
        self.assertEqual(tuned_params["objective"], "binary")
        self.assertEqual(tuned_params["subsample"], 0.75)

    def test_preserves_base_params(self):
        """Optuna tuner preserves all base params."""
        np.random.seed(42)
        n_samples = 100
        n_features = 3

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        timestamp_idx = np.arange(n_samples)

        base_params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "subsample": 0.75,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1,
        }

        tuned_params = _tune_lgbm_hyperparams_optuna(
            X_train, y_train, timestamp_idx, 10, "upper", 3, 5, base_params, n_trials=2
        )

        # All base params should be present
        for key, value in base_params.items():
            self.assertEqual(tuned_params[key], value)

    def test_respects_search_space_ranges(self):
        """Tuned params should be within expected ranges."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        timestamp_idx = np.arange(n_samples)

        base_params = {
            "objective": "binary",
            "subsample": 0.75,
            "random_state": 42,
            "verbose": -1,
        }

        tuned_params = _tune_lgbm_hyperparams_optuna(
            X_train, y_train, timestamp_idx, 10, "lower", 3, 5, base_params, n_trials=3
        )

        # Check ranges match plan specification
        self.assertGreaterEqual(tuned_params["learning_rate"], 0.01)
        self.assertLessEqual(tuned_params["learning_rate"], 0.2)

        self.assertGreaterEqual(tuned_params["max_depth"], 3)
        self.assertLessEqual(tuned_params["max_depth"], 8)

        self.assertGreaterEqual(tuned_params["n_estimators"], 100)
        self.assertLessEqual(tuned_params["n_estimators"], 400)


class MLSchemaFeatureSelectionTest(TestCase):
    """Test that MLSchema is used for feature selection."""

    def test_metadata_columns_excluded(self):
        """Metadata columns are excluded from features."""
        from quant_tick.lib.schema import MLSchema

        df_cols = [
            "timestamp",
            "bar_idx",
            "config_id",
            "entry_price",
            "close",
            "volume",
            "hit_lower_by_10",
            "hit_upper_by_10",
        ]

        features = MLSchema.get_training_features(df_cols, [10])

        # Should only have close and volume
        self.assertEqual(sorted(features), ["close", "volume"])

    def test_label_columns_excluded(self):
        """Label columns are excluded from features."""
        from quant_tick.lib.schema import MLSchema

        df_cols = [
            "close",
            "volume",
            "hit_lower_by_30",
            "hit_upper_by_30",
            "hit_lower_by_60",
            "hit_upper_by_60",
        ]

        features = MLSchema.get_training_features(df_cols, [30, 60])

        # Should only have close and volume
        self.assertEqual(sorted(features), ["close", "volume"])

    def test_config_columns_included(self):
        """Config columns like width/asymmetry are included in features."""
        from quant_tick.lib.schema import MLSchema

        df_cols = [
            "close",
            "width",
            "asymmetry",
            "range_width",
            "hit_lower_by_10",
        ]

        features = MLSchema.get_training_features(df_cols, [10])

        # Should include width, asymmetry, range_width (config cols)
        self.assertIn("width", features)
        self.assertIn("asymmetry", features)
        self.assertIn("range_width", features)
        self.assertNotIn("hit_lower_by_10", features)


class FeatureHashValidationTest(TestCase):
    """Test feature hash computation and validation."""

    def test_compute_feature_hash_is_deterministic(self):
        """Feature hash is deterministic for same feature set."""
        from quant_tick.lib.train import compute_feature_hash

        features1 = ["close", "volume", "rsi", "macd"]
        features2 = ["close", "volume", "rsi", "macd"]

        hash1 = compute_feature_hash(features1)
        hash2 = compute_feature_hash(features2)

        self.assertEqual(hash1, hash2)

    def test_compute_feature_hash_order_invariant(self):
        """Feature hash is order-invariant (uses sorted features)."""
        from quant_tick.lib.train import compute_feature_hash

        features1 = ["close", "volume", "rsi", "macd"]
        features2 = ["macd", "rsi", "volume", "close"]

        hash1 = compute_feature_hash(features1)
        hash2 = compute_feature_hash(features2)

        self.assertEqual(hash1, hash2)

    def test_compute_feature_hash_detects_changes(self):
        """Feature hash changes when features change."""
        from quant_tick.lib.train import compute_feature_hash

        features1 = ["close", "volume", "rsi"]
        features2 = ["close", "volume", "macd"]  # Different feature

        hash1 = compute_feature_hash(features1)
        hash2 = compute_feature_hash(features2)

        self.assertNotEqual(hash1, hash2)

    def test_compute_feature_hash_detects_additions(self):
        """Feature hash changes when features are added."""
        from quant_tick.lib.train import compute_feature_hash

        features1 = ["close", "volume"]
        features2 = ["close", "volume", "rsi"]  # Added feature

        hash1 = compute_feature_hash(features1)
        hash2 = compute_feature_hash(features2)

        self.assertNotEqual(hash1, hash2)

    def test_compute_feature_hash_detects_removals(self):
        """Feature hash changes when features are removed."""
        from quant_tick.lib.train import compute_feature_hash

        features1 = ["close", "volume", "rsi"]
        features2 = ["close", "volume"]  # Removed feature

        hash1 = compute_feature_hash(features1)
        hash2 = compute_feature_hash(features2)

        self.assertNotEqual(hash1, hash2)


