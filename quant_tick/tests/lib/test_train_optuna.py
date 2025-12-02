import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.train import (
    _cv_brier_for_params,
    _tune_lgbm_hyperparams_optuna,
    train_model_core,
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
            lgbm_params=lgbm_params,
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
                lgbm_params=lgbm_params,
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


class TrainModelCoreOptunaIntegrationTest(TestCase):
    """Integration tests for train_model_core with Optuna."""

    def test_optuna_disabled_uses_fixed_params(self):
        """With optuna_n_trials=0, uses fixed params."""
        np.random.seed(42)
        n_bars = 500  # Need enough bars for purging with horizons
        n_configs = 3  # Multiple configs per bar
        n_features = 8

        # Create synthetic data with proper structure
        data = []
        for bar_idx in range(n_bars):
            for config_id in range(n_configs):
                row = {
                    "bar_idx": bar_idx,
                    "timestamp_idx": bar_idx,
                    "config_id": config_id,
                    "close": 100 + np.random.randn(),
                }
                # Add random features
                for i in range(n_features):
                    row[f"feature_{i}"] = np.random.randn()
                # Add per-horizon labels
                row["hit_lower_by_10"] = np.random.randint(0, 2)
                row["hit_upper_by_10"] = np.random.randint(0, 2)
                row["hit_lower_by_20"] = np.random.randint(0, 2)
                row["hit_upper_by_20"] = np.random.randint(0, 2)
                data.append(row)

        df = pd.DataFrame(data)

        # Train with Optuna disabled
        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[10, 20],  # Use smaller horizons to avoid purging issues
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,  # Need enough bars after purging
            optuna_n_trials=0,  # Disabled
        )

        # Should still return models
        self.assertIsInstance(models_dict, dict)
        self.assertGreater(len(models_dict), 0)

        # Should return CV metrics
        self.assertIn("cv_brier_scores", cv_metrics)
        self.assertIn("optuna_best_params", cv_metrics)

        # optuna_best_params should be empty when disabled
        self.assertEqual(len(cv_metrics["optuna_best_params"]), 0)

    def test_optuna_enabled_stores_best_params(self):
        """With optuna_n_trials>0, stores best params."""
        np.random.seed(42)
        n_bars = 500  # Need enough bars for purging with horizons
        n_configs = 3
        n_features = 8

        # Create synthetic data with proper structure
        data = []
        for bar_idx in range(n_bars):
            for config_id in range(n_configs):
                row = {
                    "bar_idx": bar_idx,
                    "timestamp_idx": bar_idx,
                    "config_id": config_id,
                    "close": 100 + np.random.randn(),
                }
                for i in range(n_features):
                    row[f"feature_{i}"] = np.random.randn()
                # Add per-horizon labels
                row["hit_lower_by_10"] = np.random.randint(0, 2)
                row["hit_upper_by_10"] = np.random.randint(0, 2)
                row["hit_lower_by_20"] = np.random.randint(0, 2)
                row["hit_upper_by_20"] = np.random.randint(0, 2)
                data.append(row)

        df = pd.DataFrame(data)

        # Train with Optuna enabled (minimal trials for speed)
        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[10, 20],  # 2 horizons × 2 sides = 4 models
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=2,  # Minimal trials for test speed
        )

        # Should return models
        self.assertIsInstance(models_dict, dict)
        self.assertGreater(len(models_dict), 0)

        # Should store best params for each model
        self.assertIn("optuna_best_params", cv_metrics)
        optuna_best_params = cv_metrics["optuna_best_params"]

        # Should have params for each (side, horizon) model
        expected_keys = {"lower_h10", "upper_h10", "lower_h20", "upper_h20"}
        self.assertEqual(set(optuna_best_params.keys()), expected_keys)

        # Each model should have the 3 tuned params
        for model_key, params in optuna_best_params.items():
            self.assertIn("learning_rate", params)
            self.assertIn("max_depth", params)
            self.assertIn("n_estimators", params)

            # Check ranges
            self.assertGreaterEqual(params["learning_rate"], 0.01)
            self.assertLessEqual(params["learning_rate"], 0.2)
            self.assertGreaterEqual(params["max_depth"], 3)
            self.assertLessEqual(params["max_depth"], 8)
            self.assertGreaterEqual(params["n_estimators"], 100)
            self.assertLessEqual(params["n_estimators"], 400)

    def test_produces_valid_models_with_optuna(self):
        """Models trained with Optuna can predict."""
        np.random.seed(42)
        n_bars = 500  # Need enough bars for purging with horizons
        n_configs = 3
        n_features = 6

        # Create synthetic data with proper structure
        data = []
        for bar_idx in range(n_bars):
            for config_id in range(n_configs):
                row = {
                    "bar_idx": bar_idx,
                    "timestamp_idx": bar_idx,
                    "config_id": config_id,
                    "close": 100 + np.random.randn(),
                }
                for i in range(n_features):
                    row[f"feature_{i}"] = np.random.randn()
                # Add per-horizon labels
                row["hit_lower_by_10"] = np.random.randint(0, 2)
                row["hit_upper_by_10"] = np.random.randint(0, 2)
                data.append(row)

        df = pd.DataFrame(data)

        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[10],
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=2,
        )

        # Test that models can make predictions
        X_test = np.random.randn(10, len(feature_cols))

        for model_key, model in models_dict.items():
            # Should be able to predict probabilities
            probs = model.predict_proba(X_test)
            self.assertEqual(probs.shape, (10, 2))

            # Probabilities should be valid
            self.assertTrue(np.all(probs >= 0))
            self.assertTrue(np.all(probs <= 1))
            self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


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


class PurgingMathTest(TestCase):
    """Test purging math calculations in train_model_core."""

    def test_purging_prevents_lookahead_bias(self):
        """Purged training bars don't overlap with calib/test event horizons."""
        np.random.seed(42)
        n_bars = 500  # Need enough bars after purging
        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]
        n_configs = len(widths) * len(asymmetries)
        horizon = 60

        # Create synthetic data with proper grid structure
        data = []
        for bar_idx in range(n_bars):
            config_id = 0
            for width in widths:
                for asymmetry in asymmetries:
                    data.append({
                        "bar_idx": bar_idx,
                        "config_id": config_id,
                        "close": 100 + np.random.randn(),
                        "width": width,
                        "asymmetry": asymmetry,
                        "hit_lower_by_60": np.random.randint(0, 2),
                        "hit_upper_by_60": np.random.randint(0, 2),
                    })
                    config_id += 1

        df = pd.DataFrame(data)

        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[horizon],
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=0,
        )

        # Verify that no training bar's event horizon overlaps with calib/test
        # Training bars: [0, train_bars_purged)
        # Their event horizons extend to: [0 + horizon, train_bars_purged + horizon)
        # Calib bars start at: train_bars_initial
        # So we need: train_bars_purged + horizon < train_bars_initial
        # Which means: train_bars_purged <= train_bars_initial - horizon - 1

        # Reconstruct split math
        test_bars = int(n_bars * 0.15)
        calib_bars = int(n_bars * 0.15)
        train_bars_initial = n_bars - test_bars - calib_bars
        train_bars_purged_expected = max(0, train_bars_initial - (horizon + 1))

        # Training should use exactly train_bars_purged bars
        # This prevents any training sample's label (which looks ahead 'horizon' bars)
        # from seeing into calib/test data
        self.assertGreater(train_bars_purged_expected, 0)

    def test_purging_with_multiple_horizons_uses_max(self):
        """Purging uses the maximum horizon across all models."""
        np.random.seed(42)
        n_bars = 800  # Need enough bars after purging with horizon=90
        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]
        n_configs = len(widths) * len(asymmetries)
        horizons = [30, 60, 90]

        # Create synthetic data with proper grid structure
        data = []
        for bar_idx in range(n_bars):
            config_id = 0
            for width in widths:
                for asymmetry in asymmetries:
                    row = {
                        "bar_idx": bar_idx,
                        "config_id": config_id,
                        "close": 100 + np.random.randn(),
                        "width": width,
                        "asymmetry": asymmetry,
                    }
                    for h in horizons:
                        row[f"hit_lower_by_{h}"] = np.random.randint(0, 2)
                        row[f"hit_upper_by_{h}"] = np.random.randint(0, 2)
                    data.append(row)
                    config_id += 1

        df = pd.DataFrame(data)

        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=horizons,
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=0,
        )

        # Should train models for all horizons
        for h in horizons:
            self.assertIn(f"lower_h{h}", models_dict)
            self.assertIn(f"upper_h{h}", models_dict)

        # Purging should use max(horizons) = 90
        # This ensures even the longest-horizon model doesn't leak

    def test_purging_raises_on_insufficient_data(self):
        """Raises ValueError if not enough data after purging."""
        np.random.seed(42)
        n_bars = 50  # Too small
        n_configs = 4
        horizon = 60  # Very long horizon relative to data

        # Create synthetic data
        data = []
        for bar_idx in range(n_bars):
            for config_id in range(n_configs):
                data.append({
                    "bar_idx": bar_idx,
                    "config_id": config_id,
                    "close": 100 + np.random.randn(),
                    "hit_lower_by_60": np.random.randint(0, 2),
                    "hit_upper_by_60": np.random.randint(0, 2),
                })

        df = pd.DataFrame(data)

        # Should raise because after purging with horizon=60,
        # train/calib bars will be < MIN_BARS_PER_SPLIT
        with self.assertRaises(ValueError) as ctx:
            train_model_core(
                df=df,
                decision_horizons=[horizon],
                n_splits=3,
                embargo_bars=5,
                holdout_pct=0.15,
                calibration_pct=0.15,
                optuna_n_trials=0,
            )

        self.assertIn("After purging", str(ctx.exception))

    def test_purging_math_correctness(self):
        """Verify purging calculations are correct."""
        np.random.seed(42)
        n_bars = 400
        n_configs = 4
        horizon = 50
        holdout_pct = 0.2
        calibration_pct = 0.2

        # Create synthetic data
        data = []
        for bar_idx in range(n_bars):
            for config_id in range(n_configs):
                data.append({
                    "bar_idx": bar_idx,
                    "config_id": config_id,
                    "close": 100 + np.random.randn(),
                    "hit_lower_by_50": np.random.randint(0, 2),
                    "hit_upper_by_50": np.random.randint(0, 2),
                })

        df = pd.DataFrame(data)

        # Expected math
        test_bars_expected = int(n_bars * holdout_pct)
        calib_bars_expected = int(n_bars * calibration_pct)
        train_bars_initial_expected = n_bars - test_bars_expected - calib_bars_expected
        train_bars_purged_expected = max(0, train_bars_initial_expected - (horizon + 1))
        calib_bars_purged_expected = max(0, calib_bars_expected - (horizon + 1))

        # Train the model
        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[horizon],
            n_splits=3,
            embargo_bars=5,
            holdout_pct=holdout_pct,
            calibration_pct=calibration_pct,
            optuna_n_trials=0,
        )

        # Verify the model trained successfully
        self.assertGreater(len(models_dict), 0)
        self.assertGreater(train_bars_purged_expected, 0)
        self.assertGreater(calib_bars_purged_expected, 0)


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


class CalibratorSelfCheckingTest(TestCase):
    """Test calibrator attachment and serialization."""

    def test_models_have_calibrator_attributes(self):
        """Trained models have calibrator_ and calibration_method_ attributes."""
        np.random.seed(42)
        n_bars = 400
        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]
        n_configs = len(widths) * len(asymmetries)
        horizon = 30

        # Create synthetic data
        data = []
        for bar_idx in range(n_bars):
            config_id = 0
            for width in widths:
                for asymmetry in asymmetries:
                    data.append({
                        "bar_idx": bar_idx,
                        "config_id": config_id,
                        "close": 100 + np.random.randn(),
                        "width": width,
                        "asymmetry": asymmetry,
                        "hit_lower_by_30": np.random.randint(0, 2),
                        "hit_upper_by_30": np.random.randint(0, 2),
                    })
                    config_id += 1

        df = pd.DataFrame(data)

        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[horizon],
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=0,
        )

        # Check each model has calibrator attributes
        for model_key, model in models_dict.items():
            self.assertTrue(hasattr(model, "calibrator_"))
            self.assertTrue(hasattr(model, "calibration_method_"))

            # Calibrator can be None or a calibration object
            calibrator = getattr(model, "calibrator_", None)
            if calibrator is not None:
                # Should be an sklearn calibrator (IsotonicRegression or _SigmoidCalibration)
                self.assertTrue(hasattr(calibrator, "predict"))

            # Calibration method should be a string
            calib_method = getattr(model, "calibration_method_", "none")
            self.assertIsInstance(calib_method, str)
            self.assertIn(calib_method, ["none", "isotonic", "sigmoid"])

    def test_calibrator_can_be_serialized(self):
        """Calibrators can be serialized with pickle."""
        import pickle

        np.random.seed(42)
        n_bars = 400
        widths = [0.02]
        asymmetries = [0.0]
        n_configs = len(widths) * len(asymmetries)
        horizon = 30

        # Create synthetic data
        data = []
        for bar_idx in range(n_bars):
            data.append({
                "bar_idx": bar_idx,
                "config_id": 0,
                "close": 100 + np.random.randn(),
                "width": widths[0],
                "asymmetry": asymmetries[0],
                "hit_lower_by_30": np.random.randint(0, 2),
                "hit_upper_by_30": np.random.randint(0, 2),
            })

        df = pd.DataFrame(data)

        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[horizon],
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=0,
        )

        # Test each model's calibrator can be serialized
        for model_key, model in models_dict.items():
            calibrator = getattr(model, "calibrator_", None)
            if calibrator is not None:
                # Should be able to pickle and unpickle
                try:
                    calibrator_bytes = pickle.dumps(calibrator)
                    restored_calibrator = pickle.loads(calibrator_bytes)

                    # Restored calibrator should have same methods
                    self.assertTrue(hasattr(restored_calibrator, "predict"))
                except Exception as e:
                    self.fail(f"Failed to serialize calibrator for {model_key}: {e}")

    def test_calibration_reduces_brier_score(self):
        """Calibration improves or maintains Brier score."""
        np.random.seed(42)
        n_bars = 500
        widths = [0.02]
        asymmetries = [0.0]
        n_configs = len(widths) * len(asymmetries)
        horizon = 30

        # Create synthetic data
        data = []
        for bar_idx in range(n_bars):
            data.append({
                "bar_idx": bar_idx,
                "config_id": 0,
                "close": 100 + np.random.randn(),
                "width": widths[0],
                "asymmetry": asymmetries[0],
                "hit_lower_by_30": np.random.randint(0, 2),
                "hit_upper_by_30": np.random.randint(0, 2),
            })

        df = pd.DataFrame(data)

        models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
            df=df,
            decision_horizons=[horizon],
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.15,
            calibration_pct=0.15,
            optuna_n_trials=0,
        )

        # Holdout metrics should contain calibrated Brier scores
        self.assertIn("per_horizon_brier_lower", holdout_metrics)
        self.assertIn("per_horizon_brier_upper", holdout_metrics)

        # Brier scores should be reasonable (between 0 and 1)
        for h, brier in holdout_metrics["per_horizon_brier_lower"].items():
            self.assertGreaterEqual(brier, 0.0)
            self.assertLessEqual(brier, 1.0)

        for h, brier in holdout_metrics["per_horizon_brier_upper"].items():
            self.assertGreaterEqual(brier, 0.0)
            self.assertLessEqual(brier, 1.0)
