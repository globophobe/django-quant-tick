"""Tests for survival model training."""

import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.train import train_core


class TrainModelsTests(TestCase):
    """Tests for train_core function."""

    def test_train_core_on_synthetic(self):
        """Test training survival models on synthetic geometric survival data."""
        np.random.seed(42)
        n_bars = 500
        n_configs = 2
        max_horizon = 30

        # Generate synthetic hazard data with geometric first-touch times
        rows = []
        for bar_idx in range(n_bars):
            for cfg_id in range(n_configs):
                # Simulate first-touch time from geometric with p=0.02
                t_lower = int(np.random.geometric(0.02))
                t_upper = int(np.random.geometric(0.02))
                t_lower = min(t_lower, max_horizon + 1)
                t_upper = min(t_upper, max_horizon + 1)

                event_lower = 0 if t_lower > max_horizon else 1
                event_upper = 0 if t_upper > max_horizon else 1

                # Generate random base features
                base_features = {f"feature_{i}": np.random.randn() for i in range(5)}

                # Expand over k=1..max_horizon
                for k in range(1, max_horizon + 1):
                    rows.append({
                        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=bar_idx),
                        "bar_idx": bar_idx,
                        "config_id": cfg_id,
                        "k": k,
                        "hazard_lower": 1 if t_lower == k else 0,
                        "hazard_upper": 1 if t_upper == k else 0,
                        "event_lower": event_lower,
                        "event_upper": event_upper,
                        **base_features,
                    })

        df = pd.DataFrame(rows)

        # Train survival models
        models, feature_cols, cv_metrics, holdout_metrics = train_core(
            df=df,
            max_horizon=max_horizon,
            n_splits=3,
            embargo_bars=5,
            holdout_pct=0.2,
            calibration_pct=0.1,
            optuna_n_trials=0,  # Disable for speed
        )

        # Verify outputs
        self.assertIn("lower", models, "Should return lower model")
        self.assertIn("upper", models, "Should return upper model")
        self.assertIn("k", feature_cols, "k should be in feature columns")

        # Verify models have calibrators
        self.assertTrue(hasattr(models["lower"], "calibrator_"))
        self.assertTrue(hasattr(models["lower"], "calibration_method_"))
        self.assertTrue(hasattr(models["upper"], "calibrator_"))
        self.assertTrue(hasattr(models["upper"], "calibration_method_"))

        # Verify metrics are reasonable
        self.assertLess(holdout_metrics["avg_brier"], 0.2, "Holdout Brier should be < 0.2")
        self.assertIn("lower", holdout_metrics["base_rates"])
        self.assertIn("upper", holdout_metrics["base_rates"])

        # Base rates should be roughly 0.02 (geometric p)
        self.assertGreater(holdout_metrics["base_rates"]["lower"], 0.005)
        self.assertLess(holdout_metrics["base_rates"]["lower"], 0.05)

    def test_train_core_validates_input(self):
        """Test that train_models validates input structure."""
        # Create invalid data (missing k column)
        df = pd.DataFrame({
            "bar_idx": [0, 0],
            "config_id": [0, 1],
            "hazard_lower": [0, 1],
            "hazard_upper": [0, 0],
            "event_lower": [1, 1],
            "event_upper": [0, 0],
            "feature_1": [1.0, 2.0],
        })

        # Should raise an error due to missing k
        with self.assertRaises(Exception):
            train_core(df, max_horizon=10, n_splits=2, optuna_n_trials=0)

    def test_train_core_feature_extraction(self):
        """Test that k is properly included as a feature."""
        np.random.seed(42)
        n_bars = 200
        n_configs = 2
        max_horizon = 10

        rows = []
        for bar_idx in range(n_bars):
            for cfg_id in range(n_configs):
                t_lower = np.random.randint(1, max_horizon + 2)
                t_upper = np.random.randint(1, max_horizon + 2)

                for k in range(1, max_horizon + 1):
                    rows.append({
                        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=bar_idx),
                        "bar_idx": bar_idx,
                        "config_id": cfg_id,
                        "k": k,
                        "hazard_lower": 1 if t_lower == k else 0,
                        "hazard_upper": 1 if t_upper == k else 0,
                        "event_lower": 1 if t_lower <= max_horizon else 0,
                        "event_upper": 1 if t_upper <= max_horizon else 0,
                        "feature_a": np.random.randn(),
                        "feature_b": np.random.randn(),
                    })

        df = pd.DataFrame(rows)

        models, feature_cols, cv_metrics, holdout_metrics = train_core(
            df, max_horizon=max_horizon, n_splits=2, embargo_bars=5, optuna_n_trials=0
        )

        # k must be in feature columns
        self.assertIn("k", feature_cols)

        # Other features should also be present
        self.assertIn("feature_a", feature_cols)
        self.assertIn("feature_b", feature_cols)

        # Metadata should NOT be in features
        self.assertNotIn("bar_idx", feature_cols)
        self.assertNotIn("config_id", feature_cols)
        self.assertNotIn("timestamp", feature_cols)
        self.assertNotIn("entry_price", feature_cols)

        # Labels should NOT be in features
        self.assertNotIn("hazard_lower", feature_cols)
        self.assertNotIn("hazard_upper", feature_cols)
        self.assertNotIn("event_lower", feature_cols)
        self.assertNotIn("event_upper", feature_cols)
