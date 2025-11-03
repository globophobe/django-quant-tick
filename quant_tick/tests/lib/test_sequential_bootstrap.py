import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.ml import compute_sample_weights, sequential_bootstrap


class SequentialBootstrapTestCase(TestCase):
    """Test sequential bootstrap and formal uniqueness weighting."""

    def test_concurrency_matrix_weights(self):
        """Test concurrency matrix weighting."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h"),
            "close": [100.0] * 10,
            "event_end_idx": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        })

        df_weighted = compute_sample_weights(df)

        self.assertIn("sample_weight", df_weighted.columns)
        self.assertTrue(all(df_weighted["sample_weight"] > 0))

    def test_sequential_bootstrap_basic(self):
        """Test sequential bootstrap returns expected number of samples."""
        event_spans = [(0, 5), (3, 8), (6, 10), (9, 12), (11, 15)]
        n_samples = 100

        sampled = sequential_bootstrap(event_spans, n_samples, random_state=42)

        self.assertEqual(len(sampled), n_samples)
        self.assertTrue(all(0 <= idx < len(event_spans) for idx in sampled))

    def test_sequential_bootstrap_uniqueness(self):
        """Test sequential bootstrap produces valid samples."""
        event_spans = [(0, 5), (3, 8), (6, 10)]
        n_samples = 50

        seq_sampled = sequential_bootstrap(event_spans, n_samples, random_state=42)

        self.assertEqual(len(seq_sampled), n_samples)
        self.assertTrue(all(0 <= idx < len(event_spans) for idx in seq_sampled))

        counts = np.bincount(seq_sampled, minlength=len(event_spans))
        self.assertTrue(all(c > 0 for c in counts))

    def test_sequential_bootstrap_empty(self):
        """Test sequential bootstrap with empty events."""
        event_spans = []
        sampled = sequential_bootstrap(event_spans, 10, random_state=42)

        self.assertEqual(len(sampled), 0)

    def test_sequential_bootstrap_non_overlapping(self):
        """Test sequential bootstrap with non-overlapping events.

        Critical regression test: ensures non-overlapping events can be sampled
        after first iteration (proves static uniqueness is working correctly).

        With 200 draws and 3 disjoint events of equal length, probability that
        any event is never sampled is (2/3)^200 ≈ 10^-18 (astronomically unlikely).
        """
        event_spans = [(0, 5), (10, 15), (20, 25)]
        n_samples = 200

        sampled = sequential_bootstrap(event_spans, n_samples, random_state=42)

        self.assertEqual(len(sampled), n_samples)
        self.assertTrue(all(0 <= idx < len(event_spans) for idx in sampled))

        counts = np.bincount(sampled, minlength=len(event_spans))
        self.assertTrue(all(c > 0 for c in counts),
                        "All non-overlapping events must be sampled at least once")

    def test_concurrency_matrix_overlapping_events(self):
        """Test concurrency matrix correctly identifies overlapping events."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1h"),
            "close": [100.0] * 5,
            "event_end_idx": [3, 4, 4, 5, 6],
        })

        df_weighted = compute_sample_weights(df)

        weights = df_weighted["sample_weight"].values

        self.assertLess(weights[1], weights[4])

    def test_sequential_bootstrap_deterministic(self):
        """Test sequential bootstrap is deterministic with same random state."""
        event_spans = [(0, 5), (3, 8), (6, 10)]
        n_samples = 50

        sampled1 = sequential_bootstrap(event_spans, n_samples, random_state=42)
        sampled2 = sequential_bootstrap(event_spans, n_samples, random_state=42)

        np.testing.assert_array_equal(sampled1, sampled2)

    def test_sample_weights_no_event_end_idx(self):
        """Test sample weights default to 1.0 when event_end_idx is missing."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h"),
            "close": [100.0] * 10,
        })

        df_weighted = compute_sample_weights(df)

        self.assertTrue(all(df_weighted["sample_weight"] == 1.0))

    def test_staggered_events(self):
        """Test sample weights with staggered overlapping events."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h"),
            "close": [100.0] * 10,
            "event_end_idx": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        })

        df_weighted = compute_sample_weights(df)

        self.assertEqual(len(df_weighted), 10)
        self.assertTrue(all(df_weighted["sample_weight"] > 0))
        self.assertTrue(all(df_weighted["sample_weight"] <= 1.0))

    def test_nested_events(self):
        """Test sample weights with fully nested events."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1h"),
            "close": [100.0] * 5,
            "event_end_idx": [10, 8, 6, 4, 4],
        })

        df_weighted = compute_sample_weights(df)

        self.assertEqual(len(df_weighted), 5)
        self.assertTrue(all(df_weighted["sample_weight"] > 0))
        self.assertTrue(all(df_weighted["sample_weight"] <= 1.0))

    def test_disjoint_events(self):
        """Test sample weights with completely disjoint events."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="1h"),
            "close": [100.0] * 10,
            "event_end_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        })

        df_weighted = compute_sample_weights(df)

        self.assertEqual(len(df_weighted), 10)
        self.assertTrue(all(df_weighted["sample_weight"] == 1.0))

    def test_partial_overlap_events(self):
        """Test sample weights with partial overlaps."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="1h"),
            "close": [100.0] * 6,
            "event_end_idx": [3, 4, 5, 6, 7, 8],
        })

        df_weighted = compute_sample_weights(df)

        weights = df_weighted["sample_weight"].values

        self.assertTrue(all(w > 0 for w in weights))
        self.assertTrue(all(w <= 1.0 for w in weights))

    def test_sequential_bootstrap_staggered_events(self):
        """Test sequential bootstrap with staggered events."""
        event_spans = [(0, 5), (2, 7), (4, 9), (6, 11), (20, 25)]
        n_samples = 100

        sampled = sequential_bootstrap(event_spans, n_samples, random_state=42)

        self.assertEqual(len(sampled), n_samples)
        self.assertTrue(all(0 <= idx < len(event_spans) for idx in sampled))

    def test_sequential_bootstrap_nested_events(self):
        """Test sequential bootstrap with nested events."""
        event_spans = [(0, 20), (5, 15), (8, 12)]
        n_samples = 50

        sampled = sequential_bootstrap(event_spans, n_samples, random_state=42)
        counts = np.bincount(sampled, minlength=len(event_spans))

        self.assertEqual(len(sampled), n_samples)
        self.assertTrue(all(c > 0 for c in counts))

    def test_moderate_scale_correctness(self):
        """Test correctness with moderate number of events."""
        n_events = 100
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n_events, freq="1h"),
            "close": [100.0] * n_events,
            "event_end_idx": np.minimum(np.arange(n_events) + 10, n_events - 1),
        })

        df_weighted = compute_sample_weights(df)

        self.assertEqual(len(df_weighted), n_events)
        self.assertTrue(all(df_weighted["sample_weight"] > 0))
        self.assertTrue(all(df_weighted["sample_weight"] <= 1.0))
        self.assertTrue(np.isfinite(df_weighted["sample_weight"]).all())
