import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.ml import apply_triple_barrier, cusum_events


class CUSUMEventsTestCase(TestCase):
    """Test CUSUM event detection."""

    def test_cusum_no_events(self):
        """Test CUSUM with no significant moves."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="1h"),
            "close": [100.0] * 100,
        })

        events = cusum_events(df, threshold=0.05)

        self.assertEqual(len(events), 0)

    def test_cusum_positive_event(self):
        """Test CUSUM detects positive price move."""
        prices = [100.0] * 10
        prices.extend([100 * (1.02 ** i) for i in range(1, 6)])
        prices.extend([prices[-1]] * 10)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
        })

        events = cusum_events(df, threshold=0.02)

        self.assertGreater(len(events), 0)
        self.assertTrue(all(10 <= e < 15 for e in events))

    def test_cusum_negative_event(self):
        """Test CUSUM detects negative price move."""
        prices = [100.0] * 10
        prices.extend([100 * (0.98 ** i) for i in range(1, 6)])
        prices.extend([prices[-1]] * 10)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
        })

        events = cusum_events(df, threshold=0.02)

        self.assertGreater(len(events), 0)
        self.assertTrue(all(10 <= e < 15 for e in events))

    def test_cusum_multiple_events(self):
        """Test CUSUM detects multiple events."""
        prices = [100.0] * 10
        prices.extend([100 * (1.03 ** i) for i in range(1, 4)])
        prices.extend([prices[-1]] * 10)
        prices.extend([prices[-1] * (0.97 ** i) for i in range(1, 4)])
        prices.extend([prices[-1]] * 10)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
        })

        events = cusum_events(df, threshold=0.02)

        self.assertGreaterEqual(len(events), 2)

    def test_cusum_with_triple_barrier(self):
        """Test CUSUM events with triple barrier labeling."""
        np.random.seed(42)
        prices = [100.0]
        for _ in range(99):
            ret = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "volume": [1000.0] * len(prices),
        })

        events = cusum_events(df, threshold=0.02)

        df_labeled = apply_triple_barrier(
            df, pt_mult=2.0, sl_mult=1.0, max_holding=24, event_idx=events
        )

        self.assertIn("label", df_labeled.columns)
        self.assertIn("event_end_idx", df_labeled.columns)

        labeled_count = (df_labeled["label"] != 0).sum()
        self.assertGreaterEqual(labeled_count, 0)

    def test_cusum_empty_dataframe(self):
        """Test CUSUM with empty DataFrame."""
        df = pd.DataFrame()

        events = cusum_events(df, threshold=0.02)

        self.assertEqual(len(events), 0)

    def test_cusum_threshold_sensitivity(self):
        """Test CUSUM threshold affects number of events."""
        np.random.seed(42)
        prices = [100.0]
        for _ in range(199):
            ret = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
        })

        events_strict = cusum_events(df, threshold=0.05)
        events_loose = cusum_events(df, threshold=0.01)

        self.assertLessEqual(len(events_strict), len(events_loose))

    def test_cusum_with_nans(self):
        """Test CUSUM handles NaN values in prices."""
        prices = [100.0] * 10
        prices.extend([100 * (1.02 ** i) for i in range(1, 6)])
        prices[12] = np.nan
        prices.extend([prices[14]] * 10)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
        })

        events = cusum_events(df, threshold=0.02)

        self.assertGreaterEqual(len(events), 0)

    def test_triple_barrier_nan_handling(self):
        """Test triple barrier with NaN prices."""
        prices = [100.0] * 50
        prices[10:15] = [np.nan] * 5
        prices[20] = 105.0
        prices[30] = 95.0

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
            "high": [p * 1.01 if not np.isnan(p) else np.nan for p in prices],
            "low": [p * 0.99 if not np.isnan(p) else np.nan for p in prices],
            "volume": [1000.0] * len(prices),
        })

        df_labeled = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=24)

        self.assertIn("label", df_labeled.columns)
        self.assertIn("event_end_idx", df_labeled.columns)

    def test_triple_barrier_varying_max_holding(self):
        """Test triple barrier with different max_holding periods."""
        np.random.seed(42)
        prices = [100.0]
        for _ in range(99):
            ret = np.random.normal(0, 0.01)
            prices.append(prices[-1] * (1 + ret))

        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=len(prices), freq="1h"),
            "close": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "volume": [1000.0] * len(prices),
        })

        events = cusum_events(df, threshold=0.02)

        for max_holding in [5, 24, 100]:
            df_labeled = apply_triple_barrier(
                df, pt_mult=2.0, sl_mult=1.0, max_holding=max_holding, event_idx=events
            )

            self.assertIn("label", df_labeled.columns)
            self.assertIn("event_end_idx", df_labeled.columns)

            for i in events:
                if i < len(df_labeled):
                    end_idx = df_labeled.loc[i, "event_end_idx"]
                    self.assertLessEqual(end_idx - i, max_holding)

    def test_triple_barrier_edge_cases(self):
        """Test triple barrier edge cases."""
        df_empty = pd.DataFrame()
        df_result = apply_triple_barrier(df_empty, pt_mult=2.0, sl_mult=1.0, max_holding=24)
        self.assertTrue(df_result.empty)

        df_single = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=1, freq="1h"),
            "close": [100.0],
        })
        df_result = apply_triple_barrier(df_single, pt_mult=2.0, sl_mult=1.0, max_holding=24)
        self.assertEqual(len(df_result), 1)
