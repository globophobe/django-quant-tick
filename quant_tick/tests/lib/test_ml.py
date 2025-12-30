from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.test import TestCase
from quant_core.constants import MISSING_SENTINEL
from quant_core.features import _compute_features
from quant_core.prediction import (
    compute_bound_features,
    enforce_monotonicity,
    prepare_features,
)

from quant_tick.lib.ml import (
    PurgedKFold,
    check_position_change_allowed,
    compute_first_touch_bars,
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
        self.assertTrue(
            check_position_change_allowed(
                bars_since_last_change=20,
                min_hold_bars=15,
            )
        )

    def test_not_allowed_before_min_hold(self):
        """Not allowed before min hold bars."""
        self.assertFalse(
            check_position_change_allowed(
                bars_since_last_change=10,
                min_hold_bars=15,
            )
        )


class PurgedKFoldTest(TestCase):
    """Tests for PurgedKFold cross-validation."""

    def test_basic_split_without_event_end(self):
        """Without event_end_exclusive_idx, behaves like TimeSeriesSplit (forward-chaining)."""
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
        # Compute exclusive upper bounds (matches production: bar_idx + horizon + 1)
        event_end_exclusive_idx = np.arange(100) + horizon + 1  # [11, 12, 13, ..., 110]

        cv = PurgedKFold(n_splits=5, embargo_bars=0)
        splits = list(cv.split(X, event_end_exclusive_idx=event_end_exclusive_idx))

        # Use fold 1: TimeSeriesSplit gives test around [33:50] for 100 samples
        train_idx, test_idx = splits[1]
        test_start = test_idx.min()

        # Explicitly verify samples in [test_start - horizon - 1, test_start) are purged
        # Sample at bar i has exclusive end i+H+1; overlaps test_start if i+H+1 > test_start
        overlap_start = max(0, test_start - horizon - 1)
        for idx in range(overlap_start, test_start):
            # Sample idx has exclusive end = idx + horizon + 1
            # If idx + horizon + 1 > test_start (or >= test_start + 1), it overlaps → should be purged
            if event_end_exclusive_idx[idx] > test_start:
                self.assertNotIn(idx, train_idx,
                               f"Sample {idx} (event_end_exclusive={event_end_exclusive_idx[idx]}) "
                               f"overlaps test_start={test_start}, should be purged")

        # Verify boundary case deterministically: sample ending exactly at test_start should be KEPT
        boundary_idx = test_start - horizon - 1
        if boundary_idx >= 0:
            # This sample has event_end_exclusive = boundary_idx + horizon + 1 = test_start
            self.assertEqual(event_end_exclusive_idx[boundary_idx], test_start,
                            "Setup: boundary sample should end exactly at test_start")
            self.assertIn(boundary_idx, train_idx,
                         f"Sample {boundary_idx} (event_end_exclusive={event_end_exclusive_idx[boundary_idx]}) "
                         f"does NOT overlap test_start={test_start}, should be kept")

    def test_embargo_removes_samples_before_test(self):
        """Embargo removes training samples within N bars before test start."""
        X = np.arange(100).reshape(-1, 1)
        event_end_exclusive_idx = np.arange(100) + 1  # Events end immediately after entry bar
        embargo_bars = 10

        cv = PurgedKFold(n_splits=5, embargo_bars=embargo_bars)
        splits = list(cv.split(X, event_end_exclusive_idx=event_end_exclusive_idx))

        # TimeSeriesSplit: train is ALWAYS before test (train_idx < test_idx.min())
        # Example fold: test [20-39], train [0-9] (indices 10-19 are embargoed)
        train_idx, test_idx = splits[1]  # Use fold 1 to have train data before test
        test_start = test_idx.min()

        # Embargo zone: [test_start - embargo_bars, test_start)
        embargo_zone = set(range(max(0, test_start - embargo_bars), test_start))
        train_set = set(train_idx)

        # Verify: (1) train is before test, (2) no samples in embargo zone
        self.assertTrue(all(idx < test_start for idx in train_idx),
                       "All training samples should be before test start")
        self.assertEqual(len(train_set & embargo_zone), 0,
                        f"Training should not include samples in embargo zone {embargo_zone}")

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
        event_end_exclusive_idx = np.arange(20) + 100 + 1

        cv = PurgedKFold(n_splits=2, embargo_bars=0)
        splits = list(cv.split(X, event_end_exclusive_idx=event_end_exclusive_idx))

        # Should skip folds where purging removes all training data
        # With TimeSeriesSplit and long horizon, early folds may have no valid training
        self.assertGreaterEqual(len(splits), 0)
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)

    def test_with_bar_idx_for_interleaved_data(self):
        """Uses bar_idx for purging when provided."""
        # Simulate interleaved data: 50 bars x 3 configs = 150 rows
        n_bars = 50
        n_configs = 3
        n_rows = n_bars * n_configs
        X = np.arange(n_rows).reshape(-1, 1)

        # bar_idx repeats: [0,0,0,1,1,1,2,2,2,...]
        bar_idx = np.repeat(np.arange(n_bars), n_configs)

        # event_end_exclusive is bar_idx + horizon + 1
        horizon = 10
        event_end_exclusive_idx = bar_idx + horizon + 1

        cv = PurgedKFold(n_splits=5, embargo_bars=5)
        splits = list(
            cv.split(X, event_end_exclusive_idx=event_end_exclusive_idx, bar_idx=bar_idx)
        )

        # Some folds may be skipped if purging removes all training data
        self.assertGreater(len(splits), 0)

        for train_idx, test_idx in splits:
            test_bars = bar_idx[test_idx]
            test_start_bar = test_bars.min()

            # Check purging: all training events should end at or before test starts (no overlap)
            if len(train_idx) > 0:
                train_event_ends = event_end_exclusive_idx[train_idx]
                # For exclusive bounds, event_end_exclusive <= test_start means no overlap
                self.assertTrue(
                    np.all(train_event_ends <= test_start_bar),
                    msg=f"Some training events overlap test period. "
                    f"Max train event end: {train_event_ends.max()}, "
                    f"Test start: {test_start_bar}",
                )


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

        with self.assertLogs("quant_core.prediction", level="WARNING") as cm:
            enforce_monotonicity(horizon_probs, log_violations=True)

        # Should log that 180 was raised
        self.assertTrue(any("violation" in msg.lower() for msg in cm.output))


class BarConfigInvariantsTest(TestCase):
    """Test bar_idx and config_id invariants validation."""

    def test_valid_structure_passes(self):
        """Valid bar/config structure passes validation."""
        from quant_tick.lib.schema import MLSchema

        # Create valid structure: 3 bars × 2 configs
        df = pd.DataFrame(
            {
                "bar_idx": [0, 0, 1, 1, 2, 2],
                "config_id": [0, 1, 0, 1, 0, 1],
                "close": [100, 100, 101, 101, 102, 102],
            }
        )

        is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")

    def test_non_monotonic_bar_idx_fails(self):
        """Non-monotonic bar_idx raises error."""
        from quant_tick.lib.schema import MLSchema

        # bar_idx goes backwards
        df = pd.DataFrame(
            {
                "bar_idx": [0, 0, 2, 2, 1, 1],  # Non-monotonic
                "config_id": [0, 1, 0, 1, 0, 1],
                "close": [100, 100, 101, 101, 102, 102],
            }
        )

        is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
        self.assertFalse(is_valid)
        self.assertIn("not monotonically increasing", error_msg)

    def test_missing_config_id_fails(self):
        """Missing config_id per bar raises error."""
        from quant_tick.lib.schema import MLSchema

        # bar 1 has config_ids [0, 2] instead of [0, 1]
        df = pd.DataFrame(
            {
                "bar_idx": [0, 0, 1, 1, 2, 2],
                "config_id": [0, 1, 0, 2, 0, 1],  # Invalid: should be [0, 1] for bar 1
                "close": [100, 100, 101, 101, 102, 102],
            }
        )

        is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
        self.assertFalse(is_valid)
        self.assertIn("invalid config_ids", error_msg)
        self.assertIn("bar_idx=1", error_msg)


class LabelGenerationTests(TestCase):
    """Tests for label generation."""

    def setUp(self):
        """Create synthetic price path with known touches."""
        self.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
                "close": [100.0, 102.0, 98.0, 104.0, 96.0],
                "low": [100.0, 102.0, 98.0, 104.0, 96.0],
                "high": [100.0, 102.0, 98.0, 104.0, 96.0],
                "volume": [1000, 1100, 900, 1200, 800],
            }
        )

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

    def test_generate_labels_preserves_timestamp(self):
        """Test that generate_labels preserves timestamp, repeating per config."""
        from quant_tick.lib.ml import generate_labels

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'close': 100 + np.random.randn(100).cumsum(),
            'low': 99 + np.random.randn(100).cumsum(),
            'high': 101 + np.random.randn(100).cumsum(),
        })

        result = generate_labels(df, widths=[0.05], asymmetries=[0.0], horizons=[48])

        # Verify timestamp column exists
        self.assertIn('timestamp', result.columns, "timestamp should be preserved")

        # Verify timestamps are datetime objects (not null)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['timestamp']),
                       "timestamp should be datetime type")

        # NEW: Verify no null timestamps (critical for Position inserts)
        self.assertEqual(result['timestamp'].isna().sum(), 0,
                        "No timestamps should be null - would cause Position insert failures")

        # NEW: Verify timestamp values match input bars (accounting for dropped last bar)
        expected_timestamps = df['timestamp'].iloc[:len(df)-1].values
        n_configs = 1  # 1 width × 1 asymmetry
        # Each bar repeats n_configs times in the output
        expected_repeated = np.repeat(expected_timestamps, n_configs)
        np.testing.assert_array_equal(result['timestamp'].values, expected_repeated,
                                     err_msg="Timestamps should match input bars")

        # OLD assertions (keep for row count verification)
        expected_rows = (len(df) - 1) * n_configs
        self.assertEqual(len(result), expected_rows,
                        f"Expected {expected_rows} rows (n_bars={len(df)-1} × n_configs={n_configs})")

    def test_generate_labels_handles_ties_deterministically(self):
        """Test that ties (both bounds hit same bar) go to UP_FIRST."""
        from quant_tick.lib.ml import generate_labels

        # Deterministic 3-bar series: entry bar 0, both bounds hit at bar 1 (tie)
        # close=[100, 100, 100], width=0.06 (±3%), asymmetry=0.0
        # Bounds: lower=97, upper=103
        # Bar 1: low=97 (touches lower), high=103 (touches upper) → TIE
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1h'),
            'close': [100.0, 100.0, 100.0],
            'low': [100.0, 97.0, 100.0],    # Bar 1 touches lower bound (97)
            'high': [100.0, 103.0, 100.0],  # Bar 1 touches upper bound (103)
        })

        result = generate_labels(df, widths=[0.06], asymmetries=[0.0], horizons=[2])

        # For entry bar 0: both bounds hit at offset 1 (same bar)
        # With tie-break rule (UP_FIRST preferred), should be class 0
        # Result has (len(df)-1) × n_configs rows, so bar 0 is row 0
        label = result['first_hit_h2'].iloc[0]
        self.assertEqual(label, 0,
                        f"Tie (both bounds hit bar 1) should be UP_FIRST (0), got {label}")


class MultiExchangeCanonicalTest(TestCase):
    """Test canonical exchange selection for multi-exchange candles."""

    def test_canonical_uses_specified_exchange(self):
        """Canonical exchange matches specified exchange parameter."""
        # Create multi-exchange DataFrame with coinbase and binance
        df = pd.DataFrame(
            {
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
            }
        )

        # Compute features with coinbase as canonical
        result = _compute_features(df, canonical_exchange="coinbase")

        # Verify close column equals coinbaseClose
        self.assertTrue("close" in result.columns)
        pd.testing.assert_series_equal(
            result["close"], df["coinbaseClose"], check_names=False
        )

        # Verify binance features exist (basis, divergence, etc.)
        self.assertIn("basisBinance", result.columns)
        self.assertIn("basisPctBinance", result.columns)

    def test_canonical_uses_binance_when_specified(self):
        """Canonical can be different exchange (binance)."""
        df = pd.DataFrame(
            {
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
            }
        )

        # Compute features with binance as canonical
        result = _compute_features(df, canonical_exchange="binance")

        # Verify close column equals binanceClose
        pd.testing.assert_series_equal(
            result["close"], df["binanceClose"], check_names=False
        )

        # Verify coinbase features exist (now coinbase is "other")
        self.assertIn("basisCoinbase", result.columns)
        self.assertIn("basisPctCoinbase", result.columns)

    def test_missing_canonical_raises_error(self):
        """Raises error if canonical exchange not in data."""
        df = pd.DataFrame(
            {
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
            }
        )

        # Should raise ValueError for missing exchange
        with self.assertRaises(ValueError) as ctx:
            _compute_features(df, canonical_exchange="kraken")

        self.assertIn("kraken", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception).lower())

    def test_none_canonical_raises_error(self):
        """Raises error if canonical_exchange is None for multi-exchange."""
        df = pd.DataFrame(
            {
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
            }
        )

        # Should raise ValueError when canonical_exchange is None
        with self.assertRaises(ValueError) as ctx:
            _compute_features(df, canonical_exchange=None)

        self.assertIn("required", str(ctx.exception).lower())


class MulticlassCalibrationTest(TestCase):
    """Tests for multiclass calibration (One-vs-Rest)."""

    def test_calibrate_multiclass_ovr_happy_path(self):
        """Test multiclass calibration with synthetic data."""
        import numpy as np

        from quant_tick.lib.ml import calibrate_multiclass_ovr

        # Create synthetic 3-class data (100 samples)
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 3, n_samples)

        # Create mock predictions (slightly miscalibrated)
        y_pred_proba = np.random.dirichlet([1, 1, 1], n_samples)

        # Calibrate
        calibrators, methods = calibrate_multiclass_ovr(y_true, y_pred_proba)

        # Should return calibrators dict and methods dict
        self.assertIsNotNone(calibrators)
        self.assertIsNotNone(methods)
        self.assertEqual(len(calibrators), 3)
        self.assertEqual(len(methods), 3)

        # Each class should have a method
        for cls in range(3):
            self.assertIn(cls, calibrators)
            self.assertIn(cls, methods)

    def test_calibrate_multiclass_ovr_too_few_samples(self):
        """Test calibration skips when too few samples."""
        import numpy as np

        from quant_tick.lib.ml import calibrate_multiclass_ovr

        # Only 20 samples (< 30 threshold)
        y_true = np.array([0, 1, 2] * 6 + [0, 1])
        y_pred_proba = np.random.dirichlet([1, 1, 1], 20)

        calibrators, methods = calibrate_multiclass_ovr(y_true, y_pred_proba)

        # Should return None, None
        self.assertIsNone(calibrators)
        self.assertIsNone(methods)

    def test_calibrate_multiclass_ovr_imbalanced_class(self):
        """Test calibration skips when class has too few samples."""
        import numpy as np

        from quant_tick.lib.ml import calibrate_multiclass_ovr

        # Class 2 has only 2 samples (< 5 threshold)
        y_true = np.array([0] * 20 + [1] * 20 + [2] * 2)
        y_pred_proba = np.random.dirichlet([1, 1, 1], 42)

        calibrators, methods = calibrate_multiclass_ovr(y_true, y_pred_proba)

        # Should return None, None
        self.assertIsNone(calibrators)
        self.assertIsNone(methods)

    def test_apply_multiclass_calibration(self):
        """Test applying multiclass calibration."""
        import numpy as np
        from quant_core.prediction import apply_multiclass_calibration
        from sklearn.isotonic import IsotonicRegression

        # Create mock calibrators (identity for simplicity)
        np.random.seed(42)
        calibrators = {}
        methods = {}

        for cls in range(3):
            calibrator = IsotonicRegression(out_of_bounds="clip")
            # Fit with identity mapping
            x = np.linspace(0, 1, 100)
            calibrator.fit(x, x)
            calibrators[cls] = calibrator
            methods[cls] = "isotonic"

        # Create mock predictions
        y_pred_proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])

        # Apply calibration
        calibrated = apply_multiclass_calibration(y_pred_proba, calibrators, methods)

        # Should return probabilities that sum to 1.0
        self.assertEqual(calibrated.shape, (2, 3))
        np.testing.assert_allclose(calibrated.sum(axis=1), [1.0, 1.0], rtol=1e-5)

    def test_apply_multiclass_calibration_none(self):
        """Test calibration application with None calibrators."""
        import numpy as np
        from quant_core.prediction import apply_multiclass_calibration

        y_pred_proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])

        # Apply with None calibrators
        calibrated = apply_multiclass_calibration(y_pred_proba, None, None)

        # Should return unchanged
        np.testing.assert_array_equal(calibrated, y_pred_proba)

    def test_apply_multiclass_calibration_partial_none(self):
        """Test calibration with some classes having None calibrators."""
        import numpy as np
        from quant_core.prediction import apply_multiclass_calibration
        from sklearn.isotonic import IsotonicRegression

        # Only calibrate class 0, skip class 1 and 2
        np.random.seed(42)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        x = np.linspace(0, 1, 100)
        calibrator.fit(x, x)

        calibrators = {0: calibrator, 1: None, 2: None}
        methods = {0: "isotonic", 1: "none", 2: "none"}

        y_pred_proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])

        # Apply calibration
        calibrated = apply_multiclass_calibration(y_pred_proba, calibrators, methods)

        # Should return probabilities that sum to 1.0
        self.assertEqual(calibrated.shape, (2, 3))
        np.testing.assert_allclose(calibrated.sum(axis=1), [1.0, 1.0], rtol=1e-5)


class ClassRemappingTest(TestCase):
    """Test _remap_proba_via_classes handles non-standard class orderings."""

    def test_remap_proba_via_classes_with_non_standard_ordering(self):
        """Test class remapping with classes_ = [0, 2, 1] (non-standard order)."""
        import numpy as np

        from quant_tick.lib.train import _remap_proba_via_classes

        # Simulate model with non-standard class ordering
        class MockModel:
            classes_ = np.array([0, 2, 1])  # NOT [0, 1, 2]!

        model = MockModel()

        # Predictions from model.predict_proba() in model.classes_ order [0, 2, 1]
        # Row 0: [P(class 0)=0.5, P(class 2)=0.3, P(class 1)=0.2]
        y_pred_proba_raw = np.array([
            [0.5, 0.3, 0.2],  # In model.classes_ order: [0, 2, 1]
            [0.1, 0.4, 0.5],
        ])

        # Remap using the REAL function from train.py
        y_pred_proba_canonical = _remap_proba_via_classes(y_pred_proba_raw, model)

        # Expected: [P(class 0), P(class 1), P(class 2)]
        # Row 0: [0.5, 0.2, 0.3]  (class 1 was at index 2, class 2 was at index 1)
        # Row 1: [0.1, 0.5, 0.4]
        expected = np.array([
            [0.5, 0.2, 0.3],
            [0.1, 0.5, 0.4],
        ])

        np.testing.assert_array_almost_equal(y_pred_proba_canonical, expected,
                                            err_msg="Class remapping failed")

        # Verify probabilities still sum to 1
        np.testing.assert_allclose(y_pred_proba_canonical.sum(axis=1), [1.0, 1.0],
                                  err_msg="Probabilities should sum to 1 after remapping")

    def test_remap_proba_via_classes_no_op_when_no_classes_attr(self):
        """Test remapping is no-op when model has no classes_ attribute."""
        import numpy as np

        from quant_tick.lib.train import _remap_proba_via_classes

        class MockModelNoClasses:
            pass  # No classes_ attribute

        model = MockModelNoClasses()
        y_pred_proba = np.array([[0.5, 0.3, 0.2]])

        # Test the REAL function
        result = _remap_proba_via_classes(y_pred_proba, model)

        # Should return unchanged
        np.testing.assert_array_equal(result, y_pred_proba,
                                     err_msg="Should be no-op when no classes_ attribute")


class MLConfigValidationTest(TestCase):
    """Test MLConfig validation logic."""

    def test_inference_lookback_validation_rejects_insufficient(self):
        """Test that clean() rejects inference_lookback < max_warmup_bars."""
        from django.core.exceptions import ValidationError
        from quant_core.features import compute_max_warmup_bars

        from quant_tick.models import MLConfig

        max_warmup = compute_max_warmup_bars()

        # Create config with insufficient lookback
        cfg = MLConfig(inference_lookback=max_warmup - 1)

        # Should raise ValidationError
        with self.assertRaises(ValidationError) as ctx:
            cfg.clean()

        # Check error message mentions the field and requirement
        self.assertIn('inference_lookback', ctx.exception.message_dict)
        error_msg = ctx.exception.message_dict['inference_lookback'][0]
        self.assertIn(str(max_warmup), error_msg)
        self.assertIn('volZScore', error_msg)

    def test_inference_lookback_validation_accepts_sufficient(self):
        """Test that clean() accepts inference_lookback >= max_warmup_bars."""
        from quant_core.features import compute_max_warmup_bars

        from quant_tick.models import MLConfig

        max_warmup = compute_max_warmup_bars()

        # Create config with sufficient lookback
        cfg = MLConfig(inference_lookback=max_warmup)

        # Should not raise
        cfg.clean()

        # Also test with buffer (default value)
        cfg_default = MLConfig(inference_lookback=150)
        cfg_default.clean()

    def test_inference_lookback_default_is_sufficient(self):
        """Test that default inference_lookback is >= max_warmup_bars."""
        from quant_core.features import compute_max_warmup_bars

        from quant_tick.models import MLConfig

        cfg = MLConfig()
        max_warmup = compute_max_warmup_bars()

        # Default should be sufficient
        self.assertGreaterEqual(
            cfg.inference_lookback, max_warmup,
            f"Default inference_lookback ({cfg.inference_lookback}) should be >= "
            f"max_warmup_bars ({max_warmup})"
        )


class MLConfigHorizonsTest(TestCase):
    """Test MLConfig.get_horizons() with custom horizons configuration."""

    def test_custom_horizons_override(self):
        """Test that json_data['horizons'] overrides candle-derived horizons."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})

        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": [16, 32, 96]}
        )

        horizons = ml_config.get_horizons()
        self.assertEqual(horizons, [16, 32, 96])

    def test_custom_horizons_dedupes_and_sorts(self):
        """Test that custom horizons are deduplicated and sorted."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": [96, 16, 32, 16, 96]}
        )

        horizons = ml_config.get_horizons()
        self.assertEqual(horizons, [16, 32, 96])

    def test_custom_horizons_rejects_negative_values(self):
        """Test that negative values are rejected."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": [16, -5, 32]}
        )

        with self.assertRaises(ValueError) as ctx:
            ml_config.get_horizons()

        self.assertIn("positive integers", str(ctx.exception))

    def test_custom_horizons_rejects_zero(self):
        """Test that zero is rejected."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": [16, 0, 32]}
        )

        with self.assertRaises(ValueError) as ctx:
            ml_config.get_horizons()

        self.assertIn("positive integers", str(ctx.exception))

    def test_custom_horizons_rejects_non_list(self):
        """Test that non-list values are rejected."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": "not a list"}
        )

        with self.assertRaises(ValueError) as ctx:
            ml_config.get_horizons()

        self.assertIn("must be a list", str(ctx.exception))

    def test_custom_horizons_rejects_non_integers(self):
        """Test that non-integer values are rejected."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": [16, 32.5, 96]}
        )

        with self.assertRaises(ValueError) as ctx:
            ml_config.get_horizons()

        self.assertIn("positive integers", str(ctx.exception))

    def test_custom_horizons_rejects_booleans(self):
        """Test that booleans are rejected (isinstance(True, int) is True)."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": [16, True, 96]}
        )

        with self.assertRaises(ValueError) as ctx:
            ml_config.get_horizons()

        self.assertIn("positive integers", str(ctx.exception))

    def test_custom_horizons_rejects_empty_list(self):
        """Test that empty lists are rejected (min/max would fail downstream)."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": []}
        )

        with self.assertRaises(ValueError) as ctx:
            ml_config.get_horizons()

        self.assertIn("at least one", str(ctx.exception))

    def test_custom_horizons_accepts_tuple(self):
        """Test that tuples are accepted (for programmatic setting)."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": (16, 32, 96)}
        )

        horizons = ml_config.get_horizons()
        self.assertEqual(horizons, [16, 32, 96])

    def test_custom_horizons_none_falls_back(self):
        """Test that horizons=None falls back to candle-derived."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={"horizons": None}
        )

        horizons = ml_config.get_horizons(max_days=2)
        self.assertEqual(horizons, [24, 48])

    def test_fallback_to_candle_derived_horizons(self):
        """Test that without custom horizons, falls back to candle-derived."""
        from quant_tick.models import Candle, MLConfig

        candle = Candle(json_data={"target_candles_per_day": 24})
        ml_config = MLConfig(
            candle=candle,
            json_data={}
        )

        horizons = ml_config.get_horizons(max_days=2)
        self.assertEqual(horizons, [24, 48])
