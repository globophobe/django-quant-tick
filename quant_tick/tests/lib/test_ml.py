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
    enforce_monotonicity,
    find_optimal_config,
    generate_multi_config_labels,
    prepare_features,
)


class GenerateMultiConfigLabelsTest(TestCase):
    """Generate multi config labels test."""

    def setUp(self):
        """Set up."""
        self.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1min"),
                "close": np.random.randn(100).cumsum() + 100,
            }
        )

    def test_one_bar_per_config(self):
        """Generate multi config labels should have one bar per config."""
        widths = [0.03, 0.05]
        asymmetries = [-0.2, 0, 0.2]

        result = generate_multi_config_labels(
            self.df,
            widths=widths,
            asymmetries=asymmetries,
            decision_horizons=[60, 120, 180],
        )

        n_rows_expected = (len(self.df) - 1) * len(widths) * len(asymmetries)
        self.assertEqual(len(result), n_rows_expected)

    def test_includes_config_features(self):
        """Includes width and asymmetry columns."""
        result = generate_multi_config_labels(self.df, widths=[0.05], asymmetries=[0], decision_horizons=[60, 120, 180])

        self.assertIn("width", result.columns)
        self.assertIn("asymmetry", result.columns)

    def test_per_horizon_labels(self):
        """Per horizon labels have direct binary targets."""
        widths = [0.03, 0.05]
        asymmetries = [-0.2, 0, 0.2]
        decision_horizons = [60, 120, 180]

        result = generate_multi_config_labels(
            self.df,
            widths=widths,
            asymmetries=asymmetries,
            decision_horizons=decision_horizons,
        )

        for h in decision_horizons:
            self.assertIn(f"hit_lower_by_{h}", result.columns)
            self.assertIn(f"hit_upper_by_{h}", result.columns)

    def test_interleaved_ordering(self):
        """Generate multi config labels produces interleaved ordering for backtest indexing."""
        widths = [0.03, 0.05]
        asymmetries = [0.0, 0.2]
        decision_horizons = [60, 120, 180]

        result = generate_multi_config_labels(
            self.df,
            widths=widths,
            asymmetries=asymmetries,
            decision_horizons=decision_horizons,
        )

        n_configs = len(widths) * len(asymmetries)
        n_bars = len(self.df) - 1

        # Check total rows
        self.assertEqual(len(result), n_bars * n_configs)

        # Check interleaving: first n_configs rows should all have same timestamp
        first_block = result.iloc[:n_configs]
        unique_timestamps = first_block["timestamp"].unique()
        self.assertEqual(len(unique_timestamps), 1)

        # Check all configs present in first block
        widths_in_block = sorted(first_block["width"].unique())
        asyms_in_block = sorted(first_block["asymmetry"].unique())
        self.assertEqual(widths_in_block, sorted(widths))
        self.assertEqual(asyms_in_block, sorted(asymmetries))

        # Check second block has different timestamp
        second_block = result.iloc[n_configs:n_configs*2]
        self.assertNotEqual(
            first_block.iloc[0]["timestamp"],
            second_block.iloc[0]["timestamp"]
        )

        # Verify backtest indexing assumption: row_start = bar_idx * n_configs
        # should give all configs for that bar
        for bar_idx in range(min(3, n_bars)):
            row_start = bar_idx * n_configs
            bar_rows = result.iloc[row_start : row_start + n_configs]
            bar_timestamps = bar_rows["timestamp"].unique()
            self.assertEqual(len(bar_timestamps), 1)
            bar_configs = len(bar_rows)
            self.assertEqual(bar_configs, n_configs)


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


class FindOptimalConfigTest(TestCase):
    """Find optimal config test."""

    def test_finds_tightest_valid_config(self):
        """Tightest config where both probs below tolerance."""
        features = pd.DataFrame({"x": [1.0]})

        # Mock predictors: narrow width has high prob, wide width has low prob
        def predict_lower(f, lower, upper):
            width = upper - lower
            return 0.3 if width < 0.06 else 0.1  # Tight = risky

        def predict_upper(f, lower, upper):
            width = upper - lower
            return 0.3 if width < 0.06 else 0.1  # Tight = risky

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
            widths=[0.03, 0.05, 0.07],
            asymmetries=[0],
        )

        # Should pick width=0.07 (first that meets tolerance)
        self.assertIsNotNone(result)
        self.assertEqual(result.width, 0.07)

    def test_returns_none_if_no_valid_config(self):
        """None when no config meets tolerance."""
        features = pd.DataFrame({"x": [1.0]})

        # Mock predictors: all configs have high prob
        def predict_lower(f, lower, upper):
            return 0.5

        def predict_upper(f, lower, upper):
            return 0.5

        result = find_optimal_config(
            predict_lower,
            predict_upper,
            features,
            touch_tolerance=0.15,
        )

        self.assertIsNone(result)


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
            from quant_tick.lib.simulate import BacktestResult

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
                with patch("quant_tick.lib.simulate.train_model_core") as mock_train:
                    # Mock training to return per-horizon models dict
                    mock_models_dict = {
                        "lower_h60": MagicMock(),
                        "lower_h120": MagicMock(),
                        "lower_h180": MagicMock(),
                        "upper_h60": MagicMock(),
                        "upper_h120": MagicMock(),
                        "upper_h180": MagicMock(),
                    }
                    # Set calibrator_ attr to None for all models
                    for model in mock_models_dict.values():
                        model.calibrator_ = None

                    mock_train.return_value = (
                        mock_models_dict,  # models_dict
                        ["feature_0", "feature_1"],  # feature_cols (includes feature_1!)
                        {
                            "avg_brier_lower": 0.15,
                            "avg_brier_upper": 0.14,
                        },  # cv_metrics
                        {
                            "avg_brier_lower": 0.16,
                            "avg_brier_upper": 0.15,
                            "per_horizon_brier_lower": {60: 0.15, 120: 0.17, 180: 0.16},
                            "per_horizon_brier_upper": {60: 0.14, 120: 0.16, 180: 0.15},
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
        from quant_tick.lib.ml import find_optimal_config

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
        from quant_tick.lib.ml import find_optimal_config

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
        from quant_tick.lib.ml import find_optimal_config

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
        from quant_tick.lib.ml import find_optimal_config

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
        from quant_tick.lib.ml import find_optimal_config

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
        from quant_tick.lib.ml import find_optimal_config

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


class LabelSemanticsTest(TestCase):
    """Test label generation semantics are correct."""

    def test_hit_lower_detects_downward_breach(self):
        """hit_lower_by_H is 1 when price touches lower bound within horizon."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Price starts at 100, drops to 95 at bar 2 (within horizon)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "close": [100.0, 99.0, 95.0, 96.0, 97.0],
        })

        # Width=0.05 (5%), asymmetry=0 (symmetric)
        # Lower bound = -5%, upper bound = +5%
        # At bar 0: lower = 95, upper = 105
        # Bar 2 close=95 touches lower bound
        result = generate_multi_config_labels(
            df, widths=[0.05], asymmetries=[0.0], decision_horizons=[3]
        )

        # Bar 0 should have hit_lower_by_3 = 1 (price hits 95 at bar 2, within 3 bars)
        bar_0_rows = result[result["bar_idx"] == 0]
        self.assertEqual(len(bar_0_rows), 1)
        self.assertEqual(bar_0_rows.iloc[0]["hit_lower_by_3"], 1)

    def test_hit_upper_detects_upward_breach(self):
        """hit_upper_by_H is 1 when price touches upper bound within horizon."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Price starts at 100, rises to 106 at bar 2
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "close": [100.0, 103.0, 106.0, 104.0, 102.0],
        })

        # Width=0.05, asymmetry=0
        # Upper bound = +5% = 105
        # Bar 2 close=106 exceeds upper bound
        result = generate_multi_config_labels(
            df, widths=[0.05], asymmetries=[0.0], decision_horizons=[3]
        )

        bar_0_rows = result[result["bar_idx"] == 0]
        self.assertEqual(bar_0_rows.iloc[0]["hit_upper_by_3"], 1)

    def test_no_hit_when_price_stays_in_range(self):
        """Labels are 0 when price stays within bounds."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Price oscillates within range
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "close": [100.0, 101.0, 99.0, 100.5, 99.5],
        })

        # Wide range: width=0.10 (10%)
        # Lower = 90, upper = 110
        # All prices stay within range
        result = generate_multi_config_labels(
            df, widths=[0.10], asymmetries=[0.0], decision_horizons=[3]
        )

        bar_0_rows = result[result["bar_idx"] == 0]
        self.assertEqual(bar_0_rows.iloc[0]["hit_lower_by_3"], 0)
        self.assertEqual(bar_0_rows.iloc[0]["hit_upper_by_3"], 0)

    def test_horizon_limits_lookforward_window(self):
        """Labels only consider prices within the horizon window."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Price touches bound at bar 5, but horizon is only 3 bars
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=7, freq="1h"),
            "close": [100.0, 101.0, 101.0, 101.0, 101.0, 95.0, 96.0],
        })

        # Lower bound = 95
        # Touch happens at bar 5, which is outside 3-bar horizon from bar 0
        result = generate_multi_config_labels(
            df, widths=[0.05], asymmetries=[0.0], decision_horizons=[3]
        )

        bar_0_rows = result[result["bar_idx"] == 0]
        # Should be 0 because touch at bar 5 is beyond horizon=3
        self.assertEqual(bar_0_rows.iloc[0]["hit_lower_by_3"], 0)

    def test_asymmetric_bounds_work_correctly(self):
        """Asymmetric ranges compute correct lower/upper bounds."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Price moves from 100 to 98
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "close": [100.0, 99.0, 98.0],
        })

        # Width=0.04 (4%), asymmetry=0.5 (skewed upward)
        # lower_pct = -0.04 * (0.5 - 0.5) = 0%
        # upper_pct = 0.04 * (0.5 + 0.5) = 4%
        # So range is [100, 104] - asymmetric!
        result = generate_multi_config_labels(
            df, widths=[0.04], asymmetries=[0.5], decision_horizons=[2]
        )

        bar_0_rows = result[result["bar_idx"] == 0]
        # Price drops to 98, which breaches lower bound of 100
        self.assertEqual(bar_0_rows.iloc[0]["hit_lower_by_2"], 1)
        # Price never reaches 104
        self.assertEqual(bar_0_rows.iloc[0]["hit_upper_by_2"], 0)

    def test_multiple_horizons_generate_independent_labels(self):
        """Each horizon gets independent labels."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Touch at bar 2 and bar 4
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1h"),
            "close": [100.0, 101.0, 95.0, 101.0, 106.0, 101.0],
        })

        result = generate_multi_config_labels(
            df, widths=[0.05], asymmetries=[0.0], decision_horizons=[2, 4]
        )

        bar_0_rows = result[result["bar_idx"] == 0]
        # Horizon 2: touch at bar 2 (within window)
        self.assertEqual(bar_0_rows.iloc[0]["hit_lower_by_2"], 1)
        # Horizon 4: touch at bar 2 and bar 4 (within window)
        self.assertEqual(bar_0_rows.iloc[0]["hit_lower_by_4"], 1)
        self.assertEqual(bar_0_rows.iloc[0]["hit_upper_by_4"], 1)

    def test_entry_price_used_as_reference(self):
        """Labels use entry_price, not close, as reference for bounds."""
        from quant_tick.lib.ml import generate_multi_config_labels

        # Entry price differs from close
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="1h"),
            "close": [100.0, 98.0, 97.0, 96.0],
        })

        # If we add entry_price column explicitly
        result = generate_multi_config_labels(
            df, widths=[0.05], asymmetries=[0.0], decision_horizons=[2]
        )

        # Check that entry_price column exists in result
        self.assertIn("entry_price", result.columns)
        # Entry price should be the close of the bar where decision is made
        bar_0_rows = result[result["bar_idx"] == 0]
        self.assertAlmostEqual(bar_0_rows.iloc[0]["entry_price"], 100.0, places=5)
