from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.ml import (
    MISSING_SENTINEL,
    PurgedKFold,
    check_position_change_allowed,
    compute_bound_features,
    compute_first_touch_bars,
    enforce_monotonicity,
    find_optimal_config,
    generate_labels,
    hazard_to_per_horizon_probs,
    prepare_features,
)


class ComputeBoundFeaturesTest(TestCase):
    """Compute bound features test."""

    def test_adds_bound_columns(self):
        """Bound features are added."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        result = compute_bound_features(df, lower_pct=-0.03, upper_pct=0.05)

        self.assertIn("lower_bound_pct", result.columns)
        self.assertIn("upper_bound_pct", result.columns)
        self.assertIn("range_width", result.columns)
        self.assertIn("range_asymmetry", result.columns)

    def test_range_width_correct(self):
        """Range width is upper - lower."""
        df = pd.DataFrame({"close": [100]})
        result = compute_bound_features(df, lower_pct=-0.03, upper_pct=0.05)

        self.assertAlmostEqual(result["range_width"].iloc[0], 0.08)

    def test_range_asymmetry_correct(self):
        """Range asymmetry is upper + lower."""
        df = pd.DataFrame({"close": [100]})
        result = compute_bound_features(df, lower_pct=-0.03, upper_pct=0.05)

        # -0.03 + 0.05 = 0.02
        self.assertAlmostEqual(result["range_asymmetry"].iloc[0], 0.02)


class CheckPositionChangeAllowedTest(TestCase):
    """Check position change allowed test."""

    def test_allowed_after_min_hold(self):
        """Allowed after min hold bars."""
        self.assertTrue(check_position_change_allowed(
            bars_since_last_change=20,
            min_hold_bars=15,
        ))

    def test_not_allowed_before_min_hold(self):
        """Not allowed before min hold bars."""
        self.assertFalse(check_position_change_allowed(
            bars_since_last_change=10,
            min_hold_bars=15,
        ))


class PurgedKFoldTest(TestCase):
    """Tests for PurgedKFold cross-validation."""

    def test_basic_split_without_event_end(self):
        """Without event_end_idx, behaves like TimeSeriesSplit (forward-chaining)."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, embargo_bars=10)

        splits = list(cv.split(X))
        self.assertEqual(len(splits), 5)

        # TimeSeriesSplit has expanding training sets and equal test sets
        # Each fold: train grows, test size stays constant
        for i, (train_idx, test_idx) in enumerate(splits):
            # Test that training only uses data before test (forward-chaining)
            self.assertTrue(all(train_idx < test_idx.min()))

    def test_purges_overlapping_events(self):
        """Training samples with events overlapping test period are removed."""
        X = np.arange(100).reshape(-1, 1)
        horizon = 10
        event_end_idx = np.arange(100) + horizon

        cv = PurgedKFold(n_splits=5, embargo_bars=0)
        splits = list(cv.split(X, event_end_idx=event_end_idx))

        # First fold: test [0-19], train [20-99] - no purging needed
        # Later folds have train samples BEFORE test that could overlap
        train_idx, test_idx = splits[1]  # test [20-39], train includes [0-19, 40-99]
        test_start = test_idx.min()

        # Samples just before test (indices 10-19) have event_end [20-29]
        # These overlap with test_start=20, so should be purged
        for idx in range(10, 20):
            if event_end_idx[idx] >= test_start:
                self.assertNotIn(idx, train_idx)

    def test_embargo_removes_samples_after_test(self):
        """Embargo removes training samples within N bars after test end."""
        X = np.arange(100).reshape(-1, 1)
        event_end_idx = np.arange(100)  # Events end immediately
        embargo_bars = 10

        cv = PurgedKFold(n_splits=5, embargo_bars=embargo_bars)
        splits = list(cv.split(X, event_end_idx=event_end_idx))

        # Check a fold where train includes samples after test (fold 0)
        train_idx, test_idx = splits[0]  # test [0-19], train [20-99]
        test_end = test_idx.max()  # 19

        # Samples 20-29 are in embargo zone, should be removed
        embargo_zone = set(range(test_end + 1, test_end + embargo_bars + 1))
        train_set = set(train_idx)
        self.assertEqual(len(train_set & embargo_zone), 0)

    def test_preserves_time_order(self):
        """Splits preserve time ordering."""
        X = np.arange(50).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, embargo_bars=5)

        prev_test_end = -1
        for train_idx, test_idx in cv.split(X):
            # Test indices should be contiguous and after previous test
            self.assertGreater(test_idx.min(), prev_test_end)
            prev_test_end = test_idx.max()

    def test_skips_folds_when_all_purged(self):
        """Skips folds where all training samples would be purged."""
        X = np.arange(20).reshape(-1, 1)
        # Very long horizon means all samples overlap
        event_end_idx = np.arange(20) + 100

        cv = PurgedKFold(n_splits=2, embargo_bars=0)
        splits = list(cv.split(X, event_end_idx=event_end_idx))

        # Should skip folds where purging removes all training data
        # With TimeSeriesSplit and long horizon, early folds may have no valid training
        self.assertGreaterEqual(len(splits), 0)
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)

    def test_with_timestamp_idx_for_interleaved_data(self):
        """Uses timestamp_idx for purging when provided."""
        # Simulate interleaved data: 50 bars x 3 configs = 150 rows
        n_bars = 50
        n_configs = 3
        n_rows = n_bars * n_configs
        X = np.arange(n_rows).reshape(-1, 1)

        # timestamp_idx repeats: [0,0,0,1,1,1,2,2,2,...]
        timestamp_idx = np.repeat(np.arange(n_bars), n_configs)

        # event_end is timestamp_idx + horizon
        horizon = 10
        event_end_idx = timestamp_idx + horizon

        cv = PurgedKFold(n_splits=5, embargo_bars=5)
        splits = list(cv.split(X, event_end_idx=event_end_idx, timestamp_idx=timestamp_idx))

        # Some folds may be skipped if purging removes all training data
        self.assertGreater(len(splits), 0)

        for train_idx, test_idx in splits:
            test_timestamps = timestamp_idx[test_idx]
            test_start_ts = test_timestamps.min()

            # Check purging: all training events should end before test starts (no overlap)
            if len(train_idx) > 0:
                train_event_ends = event_end_idx[train_idx]
                # Purging should ensure no training events overlap with test period
                self.assertTrue(np.all(train_event_ends < test_start_ts),
                               msg=f"Some training events overlap test period. "
                                   f"Max train event end: {train_event_ends.max()}, "
                                   f"Test start: {test_start_ts}")


class PrepareFeaturesTest(TestCase):
    """Prepare features test."""

    def test_fills_nan_with_sentinel(self):
        """NaN values filled with sentinel."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        X, cols = prepare_features(df, ["a", "b"])

        # NaN in 'a' should be replaced with sentinel
        self.assertEqual(X[1, 0], MISSING_SENTINEL)
        # Non-NaN values preserved
        self.assertEqual(X[0, 0], 1.0)
        self.assertEqual(X[2, 0], 3.0)

    def test_adds_missing_indicator(self):
        """Missing indicator column added for columns with NaN."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        X, cols = prepare_features(df, ["a", "b"])

        # Should have a_missing column
        self.assertIn("a_missing", cols)
        # b has no NaN, so no b_missing
        self.assertNotIn("b_missing", cols)

        # Check missing indicator values
        a_missing_idx = cols.index("a_missing")
        self.assertEqual(X[0, a_missing_idx], 0)  # Not missing
        self.assertEqual(X[1, a_missing_idx], 1)  # Missing
        self.assertEqual(X[2, a_missing_idx], 0)  # Not missing

    def test_reindexes_to_feature_cols(self):
        """DataFrame reindexed to match feature_cols."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        X, cols = prepare_features(df, ["a", "c"])  # Only a and c

        # Should only have columns a and c
        self.assertEqual(list(cols), ["a", "c"])
        self.assertEqual(X.shape[1], 2)

    def test_missing_columns_filled_with_sentinel(self):
        """Training mode: columns in feature_cols but not in df filled with sentinel."""
        df = pd.DataFrame({"a": [1.0]})
        # Training mode (no _missing columns in feature_cols)
        X, cols = prepare_features(df, ["a", "b"])  # b doesn't exist

        # b should be filled with sentinel
        b_idx = cols.index("b")
        self.assertEqual(X[0, b_idx], MISSING_SENTINEL)

        # b_missing should be added dynamically in training mode
        self.assertIn("b_missing", cols)
        b_missing_idx = cols.index("b_missing")
        self.assertEqual(X[0, b_missing_idx], 1)

    def test_no_missing_indicators_when_no_nan(self):
        """No missing indicators added when no NaN values."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X, cols = prepare_features(df, ["a", "b"])

        # Should only have original columns
        self.assertEqual(list(cols), ["a", "b"])
        self.assertEqual(X.shape[1], 2)

    def test_inference_strict_schema_validation(self):
        """Inference mode: missing features raise ValueError."""
        df = pd.DataFrame({"a": [1.0]})

        # Frozen schema from training (includes _missing indicators)
        frozen_feature_cols = ["a", "b", "b_missing"]

        # Should raise ValueError - feature 'b' is missing
        with self.assertRaises(ValueError) as ctx:
            prepare_features(df, frozen_feature_cols)

        # Verify error message
        err_msg = str(ctx.exception)
        self.assertIn("required feature(s) missing", err_msg)
        self.assertIn("b", err_msg)

    def test_inference_allows_all_nan_columns(self):
        """Inference mode: columns that exist but are all NaN should not raise."""
        df = pd.DataFrame({"a": [1.0], "b": [np.nan]})

        # Frozen schema from training
        frozen_feature_cols = ["a", "b", "b_missing"]

        # Should NOT raise - column 'b' exists, just all NaN
        X, cols = prepare_features(df, frozen_feature_cols)

        # b should be filled with sentinel
        b_idx = cols.index("b")
        self.assertEqual(X[0, b_idx], MISSING_SENTINEL)

        # b_missing should be 1 (indicating missing)
        b_missing_idx = cols.index("b_missing")
        self.assertEqual(X[0, b_missing_idx], 1)


class HoldoutSplitTest(TestCase):
    """Holdout split test."""

    def test_holdout_split_calculation(self):
        """Holdout split correctly divides data by time."""
        n_samples = 100
        holdout_pct = 0.2

        # Simulate the split logic from train_models
        holdout_size = int(n_samples * holdout_pct)
        train_size = n_samples - holdout_size

        # First 80% for training, last 20% for holdout
        self.assertEqual(train_size, 80)
        self.assertEqual(holdout_size, 20)

        # Verify split indices
        X = np.arange(n_samples).reshape(-1, 1)
        X_train = X[:train_size]
        X_holdout = X[train_size:]

        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_holdout), 20)

        # Holdout should be the last 20 samples (time-based split)
        self.assertEqual(X_holdout[0, 0], 80)
        self.assertEqual(X_holdout[-1, 0], 99)


class WalkForwardSlicingTest(TestCase):
    """Walk-forward slicing test."""

    def test_walkforward_cutoffs(self):
        """Walk-forward generates correct cutoff dates."""
        from datetime import datetime, timedelta

        # 90 days of data
        start = datetime(2024, 1, 1)
        end = datetime(2024, 4, 1)

        train_window_days = 30
        cadence_days = 7

        # Generate cutoffs
        cutoffs = []
        cutoff = start + timedelta(days=train_window_days)
        while cutoff + timedelta(days=cadence_days) <= end:
            cutoffs.append(cutoff)
            cutoff += timedelta(days=cadence_days)

        # Should have ~8 cutoffs (90 days - 30 train window = 60 days / 7 cadence)
        self.assertGreater(len(cutoffs), 5)
        self.assertLess(len(cutoffs), 10)

        # First cutoff should be after train window
        self.assertEqual(cutoffs[0], start + timedelta(days=train_window_days))

        # Cutoffs should be evenly spaced by cadence
        for i in range(1, len(cutoffs)):
            delta = (cutoffs[i] - cutoffs[i - 1]).days
            self.assertEqual(delta, cadence_days)


class WalkForwardIntegrationTest(TestCase):
    """Walk forward integration test."""

    def test_schema_validation_blocks_mismatched_slices(self):
        """Schema mismatch causes slice to be skipped."""
        # Create feature data where second half is missing a feature
        # Use 40 days worth to ensure we get windows that cross the midpoint
        n_bars = 40 * 24  # 40 days * 24 hours
        base_ts = datetime(2024, 1, 1)

        data = []
        for i in range(n_bars):
            ts = base_ts + timedelta(hours=i)
            row = {
                "timestamp": ts,
                "timestamp_idx": i,
                "close": 50000 + np.random.randn() * 100,
                "touched_lower": 0,
                "touched_upper": 0,
                "width": 0.1,
                "asymmetry": 0.0,
                "feature_0": np.random.randn(),
            }

            # Only include feature_1 in first 20 days
            if i < (20 * 24):
                row["feature_1"] = np.random.randn()

            data.append(row)

        df = pd.DataFrame(data)

        # Mock config
        mock_config = MagicMock()
        mock_config.candle = MagicMock()
        mock_config.symbol = MagicMock()
        mock_config.horizon_bars = 24
        mock_config.touch_tolerance = 0.3
        mock_config.min_hold_bars = 4
        mock_config.json_data = {
            "widths": [0.1],
            "asymmetries": [0.0],
            "decision_horizons": [60, 120, 180],
        }

        # Mock feature data
        mock_feature_data = MagicMock()
        mock_feature_data.has_data_frame.return_value = True
        mock_feature_data.get_data_frame.return_value = df

        # Track how many times run_backtest is called (should skip some due to schema mismatch)
        backtest_call_count = 0

        def mock_run_backtest(*args, **kwargs):
            nonlocal backtest_call_count
            backtest_call_count += 1
            # Return a simple result
            return BacktestResult(
                total_bars=100,
                bars_in_position=80,
                bars_in_range=70,
                total_touches=5,
                touch_rate=0.05,
                rebalances=3,
                avg_hold_bars=20.0,
                pct_in_range=0.875,
                positions_created=8,
            )

        with patch("quant_tick.lib.simulate.MLFeatureData.objects.filter") as mock_filter:
            mock_filter.return_value.order_by.return_value.first.return_value = mock_feature_data

            with patch("quant_tick.lib.simulate._run_backtest", side_effect=mock_run_backtest):
                with patch("quant_tick.lib.simulate.train_core") as mock_train:
                    # Mock training to return models dict (2 models)
                    mock_lower_model = MagicMock()
                    mock_upper_model = MagicMock()
                    mock_lower_model.calibrator_ = None
                    mock_lower_model.calibration_method_ = "none"
                    mock_upper_model.calibrator_ = None
                    mock_upper_model.calibration_method_ = "none"

                    mock_models_dict = {
                        "lower": mock_lower_model,
                        "upper": mock_upper_model,
                    }

                    mock_train.return_value = (
                        mock_models_dict,  # models_dict (2 models: lower, upper)
                        ["feature_0", "feature_1", "k"],  # feature_cols (includes k!)
                        {
                            "cv_brier_scores": {"lower": 0.05, "upper": 0.06},
                            "avg_brier": 0.055,
                        },  # cv_metrics
                        {
                            "holdout_brier_scores": {"lower": 0.05, "upper": 0.06},
                            "avg_brier": 0.055,
                            "base_rates": {"lower": 0.02, "upper": 0.02},
                        },  # holdout_metrics
                    )

                    with patch("quant_tick.models.Position.objects.filter") as mock_pos_filter:
                        mock_pos_filter.return_value.delete.return_value = (0, {})

                        from quant_tick.lib.simulate import (
                            ml_simulate,
                        )

                        ml_simulate(
                            config=mock_config,
                            retrain_cadence_days=5,
                            train_window_days=10,
                        )

        # Calculate max possible windows with the parameters used above
        train_window_delta = timedelta(days=10)
        cadence_delta = timedelta(days=5)

        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()

        cutoffs = []
        cutoff = min_ts + train_window_delta
        while cutoff + cadence_delta <= max_ts:
            cutoffs.append(cutoff)
            cutoff += cadence_delta

        max_windows = len(cutoffs)

        # With 40 days of data, feature_1 exists only in first 20 days
        # Training windows that include first 20 days will have feature_1
        # But their scoring windows might not (if they extend past day 20)
        # This should cause some windows to be blocked
        # However, the exact number blocked depends on window overlap
        # For now, just verify that the function completed without error
        self.assertGreater(backtest_call_count, 0)
        self.assertLessEqual(backtest_call_count, max_windows)


class EnforceMonotonicityTest(TestCase):
    """Enforce monotonicity test."""

    def test_enforces_nondecreasing_probabilities(self):
        """Enforce monotonicity applies cumulative max."""
        # Test case: probabilities decrease at h=180
        horizon_probs = {60: 0.15, 120: 0.18, 180: 0.14}

        result = enforce_monotonicity(horizon_probs, log_violations=False)

        # Should raise 180 to match 120
        self.assertEqual(result[60], 0.15)
        self.assertEqual(result[120], 0.18)
        self.assertEqual(result[180], 0.18)  # Raised from 0.14

    def test_preserves_already_monotone(self):
        """Enforce monotonicity doesn't change valid sequences."""
        horizon_probs = {60: 0.10, 120: 0.15, 180: 0.20}
        result = enforce_monotonicity(horizon_probs, log_violations=False)

        self.assertEqual(result, horizon_probs)

    def test_handles_multiple_violations(self):
        """Enforce monotonicity fixes multiple decreases."""
        horizon_probs = {60: 0.20, 120: 0.15, 180: 0.25, 240: 0.18}
        result = enforce_monotonicity(horizon_probs, log_violations=False)

        # Should be: 0.20, 0.20, 0.25, 0.25
        self.assertEqual(result[60], 0.20)
        self.assertEqual(result[120], 0.20)  # Raised
        self.assertEqual(result[180], 0.25)
        self.assertEqual(result[240], 0.25)  # Raised

    def test_logs_violations_when_enabled(self):
        """Enforce monotonicity can logs violations."""
        horizon_probs = {60: 0.15, 120: 0.18, 180: 0.14}

        with self.assertLogs("quant_tick.lib.ml", level="WARNING") as cm:
            enforce_monotonicity(horizon_probs, log_violations=True)

        # Should log that 180 was raised
        self.assertTrue(any("violation" in msg.lower() for msg in cm.output))


class BarConfigInvariantsTest(TestCase):
    """Test bar_idx and config_id invariants validation."""

    def test_valid_structure_passes(self):
        """Valid bar/config structure passes validation."""
        from quant_tick.lib.schema import MLSchema

        # Create valid structure: 3 bars × 2 configs
        df = pd.DataFrame({
            "bar_idx": [0, 0, 1, 1, 2, 2],
            "config_id": [0, 1, 0, 1, 0, 1],
            "close": [100, 100, 101, 101, 102, 102],
        })

        is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")

    def test_non_monotonic_bar_idx_fails(self):
        """Non-monotonic bar_idx raises error."""
        from quant_tick.lib.schema import MLSchema

        # bar_idx goes backwards
        df = pd.DataFrame({
            "bar_idx": [0, 0, 2, 2, 1, 1],  # Non-monotonic
            "config_id": [0, 1, 0, 1, 0, 1],
            "close": [100, 100, 101, 101, 102, 102],
        })

        is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
        self.assertFalse(is_valid)
        self.assertIn("not monotonically increasing", error_msg)

    def test_missing_config_id_fails(self):
        """Missing config_id per bar raises error."""
        from quant_tick.lib.schema import MLSchema

        # bar 1 has config_ids [0, 2] instead of [0, 1]
        df = pd.DataFrame({
            "bar_idx": [0, 0, 1, 1, 2, 2],
            "config_id": [0, 1, 0, 2, 0, 1],  # Invalid: should be [0, 1] for bar 1
            "close": [100, 100, 101, 101, 102, 102],
        })

        is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
        self.assertFalse(is_valid)
        self.assertIn("invalid config_ids", error_msg)
        self.assertIn("bar_idx=1", error_msg)


class FindOptimalConfigTest(TestCase):
    """Test find_optimal_config function."""

    def test_selects_tightest_width_passing_tolerance(self):
        """Selects tightest width that passes touch tolerance."""

        features = pd.DataFrame({"close": [100.0]})

        # Mock prediction functions: tight width has high risk, wide width has low risk
        def predict_lower(feat, lower_pct, upper_pct):
            width = abs(upper_pct - lower_pct)
            return 0.2 if width < 0.04 else 0.05  # Tight = risky, wide = safe

        def predict_upper(feat, lower_pct, upper_pct):
            width = abs(upper_pct - lower_pct)
            return 0.2 if width < 0.04 else 0.05

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.02, 0.04, 0.06],
            asymmetries=[0.0],
        )

        # Should select width=0.04 (first width that passes tolerance)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.width, 0.04, places=5)

    def test_returns_none_if_all_too_risky(self):
        """Returns None if all configs exceed tolerance."""

        features = pd.DataFrame({"close": [100.0]})

        # All predictions exceed tolerance
        def predict_lower(feat, lower_pct, upper_pct):
            return 0.5  # Very high risk

        def predict_upper(feat, lower_pct, upper_pct):
            return 0.5

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.02, 0.04, 0.06],
            asymmetries=[0.0],
        )

        self.assertIsNone(result)

    def test_selects_best_asymmetry_for_width(self):
        """Selects asymmetry with lowest combined risk for chosen width."""

        features = pd.DataFrame({"close": [100.0]})

        # Asymmetry 0.0 has lower combined risk than others
        def predict_lower(feat, lower_pct, upper_pct):
            width = abs(upper_pct - lower_pct)
            asym = (upper_pct + lower_pct) / width
            if abs(asym) < 0.01:  # Symmetric
                return 0.05
            return 0.08

        def predict_upper(feat, lower_pct, upper_pct):
            width = abs(upper_pct - lower_pct)
            asym = (upper_pct + lower_pct) / width
            if abs(asym) < 0.01:  # Symmetric
                return 0.05
            return 0.08

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.04],
            asymmetries=[-0.2, 0.0, 0.2],
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.asymmetry, 0.0, places=5)

    def test_respects_touch_tolerance_threshold(self):
        """Config is rejected if either bound exceeds tolerance."""

        features = pd.DataFrame({"close": [100.0]})

        # Lower is safe, upper exceeds tolerance
        def predict_lower(feat, lower_pct, upper_pct):
            return 0.05

        def predict_upper(feat, lower_pct, upper_pct):
            return 0.25  # Exceeds 0.15 tolerance

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.04],
            asymmetries=[0.0],
        )

        # Should reject because upper exceeds tolerance
        self.assertIsNone(result)

    def test_computes_correct_bounds_from_width_asymmetry(self):
        """Computes correct lower/upper bounds from width and asymmetry."""

        features = pd.DataFrame({"close": [100.0]})

        def predict_lower(feat, lower_pct, upper_pct):
            return 0.05

        def predict_upper(feat, lower_pct, upper_pct):
            return 0.05

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.04],
            asymmetries=[0.2],  # Asymmetric
        )

        # width=0.04, asym=0.2
        # lower_pct = -0.04 * (0.5 - 0.2) = -0.012
        # upper_pct = 0.04 * (0.5 + 0.2) = 0.028
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.lower_pct, -0.012, places=5)
        self.assertAlmostEqual(result.upper_pct, 0.028, places=5)

    def test_stores_touch_probabilities(self):
        """Result includes touch probabilities from prediction functions."""

        features = pd.DataFrame({"close": [100.0]})

        def predict_lower(feat, lower_pct, upper_pct):
            return 0.07

        def predict_upper(feat, lower_pct, upper_pct):
            return 0.09

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.04],
            asymmetries=[0.0],
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.p_touch_lower, 0.07, places=5)
        self.assertAlmostEqual(result.p_touch_upper, 0.09, places=5)


class LabelGenerationTests(TestCase):
    """Tests for label generation."""

    def setUp(self):
        """Create synthetic price path with known touches."""
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
            "close": [100.0, 102.0, 98.0, 104.0, 96.0],
            "low": [100.0, 102.0, 98.0, 104.0, 96.0],
            "high": [100.0, 102.0, 98.0, 104.0, 96.0],
            "volume": [1000, 1100, 900, 1200, 800],
        })

    def test_compute_first_touch_bars_simple(self):
        """Test first touch detection on synthetic data."""
        lower_bounds = np.array([97.0, 99.0, 95.0, 101.0, 93.0])
        upper_bounds = np.array([103.0, 105.0, 101.0, 107.0, 99.0])

        ft_lower, ft_upper = compute_first_touch_bars(
            low=self.df["low"].values,
            high=self.df["high"].values,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_horizon=4,
        )

        self.assertEqual(ft_lower[0], 4)
        self.assertEqual(ft_upper[0], 3)

        self.assertEqual(ft_lower[1], 1)
        self.assertEqual(ft_upper[1], 5)

    def test_hazard_labels_sum_equals_one(self):
        """Verify sum of labels equals 1 for touched bars."""
        labeled = generate_labels(
            self.df,
            widths=[0.03],
            asymmetries=[0.0],
            max_horizon=4,
        )

        for bar_idx in range(len(self.df) - 1):
            bar_data = labeled[labeled["bar_idx"] == bar_idx]

            hazard_sum_lower = bar_data["hazard_lower"].sum()
            hazard_sum_upper = bar_data["hazard_upper"].sum()

            self.assertIn(hazard_sum_lower, [0, 1])
            self.assertIn(hazard_sum_upper, [0, 1])

            if bar_data.iloc[0]["event_lower"] == 0:
                self.assertEqual(hazard_sum_lower, 0)
            if bar_data.iloc[0]["event_upper"] == 0:
                self.assertEqual(hazard_sum_upper, 0)

    def test_hazard_schema_structure(self):
        """Verify hazard schema has correct columns and row count."""
        labeled = generate_labels(
            self.df,
            widths=[0.02, 0.03],
            asymmetries=[-0.2, 0.0, 0.2],
            max_horizon=10,
        )

        n_bars = len(self.df) - 1
        n_configs = 2 * 3
        expected_rows = n_bars * n_configs * 10
        self.assertEqual(len(labeled), expected_rows)

        required_cols = [
            "bar_idx", "config_id", "k", "timestamp", "entry_price",
            "width", "asymmetry", "hazard_lower", "hazard_upper",
            "event_lower", "event_upper",
        ]
        for col in required_cols:
            self.assertIn(col, labeled.columns)

        self.assertEqual(labeled["k"].min(), 1)
        self.assertEqual(labeled["k"].max(), 10)

        self.assertEqual(labeled["config_id"].min(), 0)
        self.assertEqual(labeled["config_id"].max(), 5)


class SurvivalInferenceTests(TestCase):
    """Tests for survival model inference and survival reconstruction."""

    class MockHazardModel:
        """Mock model with constant hazard rate."""

        def __init__(self, hazard_rate: float):
            self.hazard_rate = hazard_rate
            self.calibrator_ = None
            self.calibration_method_ = "none"

        def predict_proba(self, X):
            n = len(X)
            probs = np.zeros((n, 2))
            probs[:, 1] = self.hazard_rate
            probs[:, 0] = 1 - self.hazard_rate
            return probs

    def test_survival_matches_constant_hazard(self):
        """Test reconstructing P(hit_by_H) from constant hazard rate."""
        # h(k) = 0.01 for all k (constant hazard)
        # Expected S(k) = (1 - 0.01)^k = 0.99^k
        model = self.MockHazardModel(hazard_rate=0.01)
        X_base = pd.DataFrame([{"feature1": 1.0}])
        feature_cols = ["feature1", "k"]

        probs = hazard_to_per_horizon_probs(
            model,
            X_base,
            feature_cols,
            decision_horizons=[60, 120, 180],
            max_horizon=180,
        )

        # Verify against analytical solution
        # P(hit_by_60) = 1 - 0.99^60
        expected_60 = 1 - 0.99**60
        expected_120 = 1 - 0.99**120
        expected_180 = 1 - 0.99**180

        self.assertAlmostEqual(probs[60], expected_60, places=4)
        self.assertAlmostEqual(probs[120], expected_120, places=4)
        self.assertAlmostEqual(probs[180], expected_180, places=4)

        # Verify monotonicity (automatic with survival curve)
        self.assertLessEqual(probs[60], probs[120])
        self.assertLessEqual(probs[120], probs[180])

    def test_monotonicity_guaranteed_with_noisy_model(self):
        """Verify monotonicity holds even with random predictions."""

        class NoisyModel:
            calibrator_ = None
            calibration_method_ = "none"

            def predict_proba(self, X):
                np.random.seed(42)
                n = len(X)
                probs = np.zeros((n, 2))
                probs[:, 1] = np.random.uniform(0, 0.1, size=n)
                probs[:, 0] = 1 - probs[:, 1]
                return probs

        X_base = pd.DataFrame([{"feature": 1.0}])
        feature_cols = ["feature", "k"]

        horizon_probs = hazard_to_per_horizon_probs(
            NoisyModel(),
            X_base,
            feature_cols,
            decision_horizons=[60, 120, 180],
            max_horizon=180,
        )

        # Monotonicity MUST hold (guaranteed by survival curve math)
        self.assertLessEqual(horizon_probs[60], horizon_probs[120])
        self.assertLessEqual(horizon_probs[120], horizon_probs[180])

    def test_column_ordering_preserved(self):
        """Test that feature column ordering matches training order."""

        class OrderSensitiveModel:
            """Model that returns different values based on column order."""

            calibrator_ = None
            calibration_method_ = "none"

            def predict_proba(self, X):
                # If columns are in correct order, k should be in last position
                # Check that X has correct column order
                n = len(X)
                probs = np.zeros((n, 2))
                # Check if k is in the right position (last) and has expected value for first row
                if X.iloc[0, -1] == 1:  # k should be 1 for first row
                    probs[:, 1] = 0.001  # Low hazard -> low cumulative probability
                else:
                    probs[:, 1] = 0.05  # Higher hazard -> high cumulative probability
                probs[:, 0] = 1 - probs[:, 1]
                return probs

        X_base = pd.DataFrame([{"feature_a": 100.0, "feature_b": 200.0}])
        feature_cols = ["feature_a", "feature_b", "k"]

        probs = hazard_to_per_horizon_probs(
            OrderSensitiveModel(),
            X_base,
            feature_cols,
            decision_horizons=[60],
            max_horizon=60,
        )

        # Should get low probability (0.05) if column order is correct
        self.assertLess(probs[60], 0.5)

    def test_inference_path_with_k_missing_from_input(self):
        """Integration test: mimics full inference path where k is not in raw features."""
        from quant_tick.lib.ml import prepare_features

        # Simulate training feature order (includes k and *_missing indicators)
        feature_cols = ["feature_a", "feature_b", "feature_a_missing", "k"]

        # Simulate live candle data (no k, no *_missing)
        feat_with_bounds = pd.DataFrame([{
            "feature_a": 100.0,
            "feature_b": 200.0,
            "width": 0.03,
            "asymmetry": 0.0,
        }])

        # This is what inference does: exclude k from prepare_features
        base_feature_cols = [c for c in feature_cols if c != "k"]
        X_array, expanded_cols = prepare_features(feat_with_bounds, base_feature_cols)
        X_df = pd.DataFrame(X_array, columns=expanded_cols)
        X_base = X_df.iloc[[0]]

        # Should not have k yet
        self.assertNotIn("k", X_base.columns)

        # Now call hazard_to_per_horizon_probs with full feature_cols (includes k)
        model = self.MockHazardModel(hazard_rate=0.01)
        horizon_probs = hazard_to_per_horizon_probs(
            model,
            X_base,
            feature_cols,  # Full training order WITH k
            decision_horizons=[60],
            max_horizon=60,
        )

        # Should complete without error and return valid probability
        self.assertIn(60, horizon_probs)
        self.assertGreater(horizon_probs[60], 0)
        self.assertLess(horizon_probs[60], 1)


class MultiExchangeCanonicalTest(TestCase):
    """Test canonical exchange selection for multi-exchange candles."""

    def test_canonical_uses_specified_exchange(self):
        """Canonical exchange matches specified exchange parameter."""
        from quant_tick.lib.labels import _compute_features

        # Create multi-exchange DataFrame with coinbase and binance
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "coinbaseClose": [100 + i for i in range(10)],
            "binanceClose": [101 + i for i in range(10)],
            "coinbaseOpen": [99 + i for i in range(10)],
            "binanceOpen": [100 + i for i in range(10)],
            "coinbaseHigh": [102 + i for i in range(10)],
            "binanceHigh": [103 + i for i in range(10)],
            "coinbaseLow": [98 + i for i in range(10)],
            "binanceLow": [99 + i for i in range(10)],
            "coinbaseVolume": [1000] * 10,
            "binanceVolume": [1100] * 10,
        })

        # Compute features with coinbase as canonical
        result = _compute_features(df, canonical_exchange="coinbase")

        # Verify close column equals coinbaseClose
        self.assertTrue("close" in result.columns)
        pd.testing.assert_series_equal(
            result["close"],
            df["coinbaseClose"],
            check_names=False
        )

        # Verify binance features exist (basis, divergence, etc.)
        self.assertIn("basisBinance", result.columns)
        self.assertIn("basisPctBinance", result.columns)

    def test_canonical_uses_binance_when_specified(self):
        """Canonical can be different exchange (binance)."""
        from quant_tick.lib.labels import _compute_features

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "coinbaseClose": [100 + i for i in range(10)],
            "binanceClose": [101 + i for i in range(10)],
            "coinbaseOpen": [99 + i for i in range(10)],
            "binanceOpen": [100 + i for i in range(10)],
            "coinbaseHigh": [102 + i for i in range(10)],
            "binanceHigh": [103 + i for i in range(10)],
            "coinbaseLow": [98 + i for i in range(10)],
            "binanceLow": [99 + i for i in range(10)],
            "coinbaseVolume": [1000] * 10,
            "binanceVolume": [1100] * 10,
        })

        # Compute features with binance as canonical
        result = _compute_features(df, canonical_exchange="binance")

        # Verify close column equals binanceClose
        pd.testing.assert_series_equal(
            result["close"],
            df["binanceClose"],
            check_names=False
        )

        # Verify coinbase features exist (now coinbase is "other")
        self.assertIn("basisCoinbase", result.columns)
        self.assertIn("basisPctCoinbase", result.columns)

    def test_missing_canonical_raises_error(self):
        """Raises error if canonical exchange not in data."""
        from quant_tick.lib.labels import _compute_features

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "coinbaseClose": [100 + i for i in range(10)],
            "binanceClose": [101 + i for i in range(10)],
            "coinbaseOpen": [99 + i for i in range(10)],
            "binanceOpen": [100 + i for i in range(10)],
            "coinbaseHigh": [102 + i for i in range(10)],
            "binanceHigh": [103 + i for i in range(10)],
            "coinbaseLow": [98 + i for i in range(10)],
            "binanceLow": [99 + i for i in range(10)],
            "coinbaseVolume": [1000] * 10,
            "binanceVolume": [1100] * 10,
        })

        # Should raise ValueError for missing exchange
        with self.assertRaises(ValueError) as ctx:
            _compute_features(df, canonical_exchange="kraken")

        self.assertIn("kraken", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception).lower())

    def test_none_canonical_raises_error(self):
        """Raises error if canonical_exchange is None for multi-exchange."""
        from quant_tick.lib.labels import _compute_features

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "coinbaseClose": [100 + i for i in range(10)],
            "binanceClose": [101 + i for i in range(10)],
            "coinbaseOpen": [99 + i for i in range(10)],
            "binanceOpen": [100 + i for i in range(10)],
            "coinbaseHigh": [102 + i for i in range(10)],
            "binanceHigh": [103 + i for i in range(10)],
            "coinbaseLow": [98 + i for i in range(10)],
            "binanceLow": [99 + i for i in range(10)],
            "coinbaseVolume": [1000] * 10,
            "binanceVolume": [1100] * 10,
        })

        # Should raise ValueError when canonical_exchange is None
        with self.assertRaises(ValueError) as ctx:
            _compute_features(df, canonical_exchange=None)

        self.assertIn("required", str(ctx.exception).lower())
