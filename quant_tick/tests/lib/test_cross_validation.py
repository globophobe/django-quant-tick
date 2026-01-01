import numpy as np
from django.test import SimpleTestCase

from quant_tick.lib.cross_validation import PurgedKFold


class PurgedKFoldTest(SimpleTestCase):
    """PurgedKFold cross-validation tests."""

    def test_basic_split_without_event_end(self):
        """Without event_end_exclusive_idx, behaves like TimeSeriesSplit."""
        X = np.arange(100).reshape(-1, 1)
        bar_idx = np.arange(len(X))
        cv = PurgedKFold(n_splits=5, embargo_bars=10)

        splits = list(cv.split(X, bar_idx=bar_idx))
        self.assertEqual(len(splits), 5)

        for train_idx, test_idx in splits:
            self.assertTrue(all(train_idx < test_idx.min()))

    def test_purges_overlapping_events(self):
        """Training samples overlapping test period are removed."""
        X = np.arange(100).reshape(-1, 1)
        bar_idx = np.arange(len(X))
        horizon = 10
        event_end_exclusive_idx = np.arange(100) + horizon + 1

        cv = PurgedKFold(n_splits=5, embargo_bars=0)
        splits = list(
            cv.split(
                X, event_end_exclusive_idx=event_end_exclusive_idx, bar_idx=bar_idx
            )
        )

        train_idx, test_idx = splits[1]
        test_start = test_idx.min()

        overlap_start = max(0, test_start - horizon - 1)
        for idx in range(overlap_start, test_start):
            if event_end_exclusive_idx[idx] > test_start:
                self.assertNotIn(idx, train_idx)

        boundary_idx = test_start - horizon - 1
        if boundary_idx >= 0:
            self.assertEqual(event_end_exclusive_idx[boundary_idx], test_start)
            self.assertIn(boundary_idx, train_idx)

    def test_embargo_removes_samples_before_test(self):
        """Embargo removes training samples within N bars before test start."""
        X = np.arange(100).reshape(-1, 1)
        bar_idx = np.arange(len(X))
        event_end_exclusive_idx = np.arange(100) + 1
        embargo_bars = 10

        cv = PurgedKFold(n_splits=5, embargo_bars=embargo_bars)
        splits = list(
            cv.split(
                X, event_end_exclusive_idx=event_end_exclusive_idx, bar_idx=bar_idx
            )
        )

        train_idx, test_idx = splits[1]
        test_start = test_idx.min()

        embargo_zone = set(range(max(0, test_start - embargo_bars), test_start))
        train_set = set(train_idx)

        self.assertTrue(all(idx < test_start for idx in train_idx))
        self.assertEqual(len(train_set & embargo_zone), 0)

    def test_preserves_time_order(self):
        """Splits preserve time ordering."""
        X = np.arange(50).reshape(-1, 1)
        bar_idx = np.arange(len(X))
        cv = PurgedKFold(n_splits=5, embargo_bars=5)

        prev_test_end = -1
        for _, test_idx in cv.split(X, bar_idx=bar_idx):
            self.assertGreater(test_idx.min(), prev_test_end)
            prev_test_end = test_idx.max()

    def test_skips_folds_when_all_purged(self):
        """Skips folds where all training samples would be purged."""
        X = np.arange(20).reshape(-1, 1)
        bar_idx = np.arange(len(X))
        event_end_exclusive_idx = np.arange(20) + 100 + 1

        cv = PurgedKFold(n_splits=2, embargo_bars=0)
        splits = list(
            cv.split(
                X, event_end_exclusive_idx=event_end_exclusive_idx, bar_idx=bar_idx
            )
        )

        self.assertGreaterEqual(len(splits), 0)
        for train_idx, _ in splits:
            self.assertGreater(len(train_idx), 0)

    def test_with_bar_idx_for_interleaved_data(self):
        """Uses bar_idx for purging when provided."""
        n_bars = 50
        n_configs = 3
        n_rows = n_bars * n_configs
        X = np.arange(n_rows).reshape(-1, 1)

        bar_idx = np.repeat(np.arange(n_bars), n_configs)
        horizon = 10
        event_end_exclusive_idx = bar_idx + horizon + 1

        cv = PurgedKFold(n_splits=5, embargo_bars=5)
        splits = list(
            cv.split(
                X, event_end_exclusive_idx=event_end_exclusive_idx, bar_idx=bar_idx
            )
        )

        self.assertGreater(len(splits), 0)

        for train_idx, test_idx in splits:
            test_bars = bar_idx[test_idx]
            test_start_bar = test_bars.min()
            if len(train_idx) > 0:
                train_event_ends = event_end_exclusive_idx[train_idx]
                self.assertTrue(np.all(train_event_ends <= test_start_bar))
