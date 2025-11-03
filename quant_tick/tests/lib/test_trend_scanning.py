"""Tests for trend scanning library (AFML Chapter 15)."""
import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.lib.trend_scanning import (
    compute_trend_statistic,
    detect_break,
    generate_trend_windows,
    rank_trends,
    scan_trends,
)


class TestGenerateTrendWindows(TestCase):
    """Tests for generate_trend_windows."""

    def test_basic_window_generation(self):
        """Generate windows with single window size."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1h')
        })

        windows = generate_trend_windows(df, window_sizes=[100], step=50, min_obs=30)

        self.assertEqual(len(windows), 19)
        self.assertEqual(windows[0]['start_idx'], 0)
        self.assertEqual(windows[0]['end_idx'], 100)
        self.assertEqual(windows[0]['size'], 100)
        self.assertEqual(windows[1]['start_idx'], 50)
        self.assertEqual(windows[1]['end_idx'], 150)

    def test_multiple_window_sizes(self):
        """Generate windows with multiple sizes."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1h')
        })

        windows = generate_trend_windows(
            df,
            window_sizes=[100, 250, 500],
            step=100,
            min_obs=50
        )

        sizes = [w['size'] for w in windows]
        self.assertIn(100, sizes)
        self.assertIn(250, sizes)
        self.assertIn(500, sizes)

        size_100_windows = [w for w in windows if w['size'] == 100]
        self.assertEqual(len(size_100_windows), 10)

    def test_min_obs_filtering(self):
        """Filter out windows below min_obs."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h')
        })

        windows = generate_trend_windows(
            df,
            window_sizes=[20, 50, 200],
            step=10,
            min_obs=30
        )

        sizes = [w['size'] for w in windows]
        self.assertNotIn(20, sizes)
        self.assertIn(50, sizes)
        self.assertNotIn(200, sizes)

    def test_timestamp_metadata(self):
        """Verify timestamp metadata in windows."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=500, freq='1h')
        })

        windows = generate_trend_windows(df, window_sizes=[100], step=100, min_obs=30)

        first_window = windows[0]
        self.assertEqual(first_window['timestamp_start'], df.iloc[0]['timestamp'])
        self.assertEqual(first_window['timestamp_end'], df.iloc[99]['timestamp'])


class TestComputeTrendStatistic(TestCase):
    """Tests for compute_trend_statistic."""

    def test_sharpe_ratio_positive_trend(self):
        """Compute Sharpe for positive trend."""
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018] * 20)

        result = compute_trend_statistic(returns, method='sharpe')

        self.assertGreater(result['score'], 0)
        self.assertGreater(result['mean'], 0)
        self.assertGreater(result['std'], 0)
        self.assertLess(result['p_value'], 0.05)

    def test_sharpe_ratio_negative_trend(self):
        """Compute Sharpe for negative trend."""
        returns = np.array([-0.01, -0.02, -0.015, -0.012, -0.018] * 20)

        result = compute_trend_statistic(returns, method='sharpe')

        self.assertLess(result['score'], 0)
        self.assertLess(result['mean'], 0)
        self.assertGreater(result['std'], 0)

    def test_t_statistic(self):
        """Compute t-statistic."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 100)

        result = compute_trend_statistic(returns, method='t_stat')

        self.assertIn('score', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('p_value', result)

    def test_weighted_statistics(self):
        """Compute statistics with sample weights."""
        returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])
        weights = np.array([1.0, 0.5, 0.8, 0.6, 0.9])

        result_weighted = compute_trend_statistic(returns, weights=weights)
        result_unweighted = compute_trend_statistic(returns)

        self.assertNotEqual(result_weighted['mean'], result_unweighted['mean'])

    def test_zero_std_handling(self):
        """Handle zero standard deviation."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])

        result = compute_trend_statistic(returns)

        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['std'], 0.0)
        self.assertEqual(result['p_value'], 1.0)

    def test_empty_returns(self):
        """Handle empty returns array."""
        returns = np.array([])

        result = compute_trend_statistic(returns)

        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['mean'], 0.0)
        self.assertEqual(result['std'], 0.0)
        self.assertEqual(result['p_value'], 1.0)


class TestScanTrends(TestCase):
    """Tests for scan_trends."""

    def test_basic_scan(self):
        """Scan trends without purging."""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 500)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=500, freq='1h')
        })
        windows = generate_trend_windows(df, window_sizes=[100, 250], step=100, min_obs=50)

        results = scan_trends(
            returns,
            weights=None,
            event_end_idx=None,
            windows=windows,
            embargo_bars=0,
            method='sharpe'
        )

        self.assertEqual(len(results), len(windows))
        for result in results:
            self.assertIn('window', result)
            self.assertIn('statistic', result)
            self.assertIn('n_events', result)
            self.assertGreater(result['n_events'], 0)

    def test_scan_with_purging(self):
        """Scan trends with purging."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0.01, 0.02, n)
        event_end_idx = np.minimum(np.arange(n) + 10, n - 1)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h')
        })
        windows = generate_trend_windows(df, window_sizes=[100], step=100, min_obs=50)

        results = scan_trends(
            returns,
            weights=None,
            event_end_idx=event_end_idx,
            windows=windows,
            embargo_bars=5,
            method='sharpe'
        )

        for result in results:
            window_size = result['window']['size']
            self.assertLessEqual(result['n_events'], window_size)

    def test_scan_with_weights(self):
        """Scan trends with sample weights."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0.01, 0.02, n)
        weights = np.random.uniform(0.5, 1.0, n)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h')
        })
        windows = generate_trend_windows(df, window_sizes=[100], step=100, min_obs=50)

        results_weighted = scan_trends(returns, weights, None, windows, 0, 'sharpe')
        results_unweighted = scan_trends(returns, None, None, windows, 0, 'sharpe')

        for w_res, u_res in zip(results_weighted, results_unweighted, strict=False):
            self.assertNotEqual(
                w_res['statistic']['score'],
                u_res['statistic']['score']
            )


class TestRankTrends(TestCase):
    """Tests for rank_trends."""

    def test_rank_by_score(self):
        """Rank trends by absolute score."""
        scan_results = [
            {'statistic': {'score': 1.5}, 'window': {}, 'n_events': 100},
            {'statistic': {'score': -2.8}, 'window': {}, 'n_events': 100},
            {'statistic': {'score': 0.5}, 'window': {}, 'n_events': 100},
            {'statistic': {'score': 3.2}, 'window': {}, 'n_events': 100},
        ]

        top_trends = rank_trends(scan_results, top_k=2)

        self.assertEqual(len(top_trends), 2)
        self.assertEqual(top_trends[0]['statistic']['score'], 3.2)
        self.assertEqual(top_trends[1]['statistic']['score'], -2.8)

    def test_rank_top_k_limit(self):
        """Respect top_k limit."""
        scan_results = [
            {'statistic': {'score': i * 0.5}, 'window': {}, 'n_events': 100}
            for i in range(20)
        ]

        top_10 = rank_trends(scan_results, top_k=10)
        top_5 = rank_trends(scan_results, top_k=5)

        self.assertEqual(len(top_10), 10)
        self.assertEqual(len(top_5), 5)

    def test_rank_empty_results(self):
        """Handle empty scan results."""
        top_trends = rank_trends([], top_k=10)

        self.assertEqual(len(top_trends), 0)


class TestDetectBreak(TestCase):
    """Tests for detect_break."""

    def test_first_scan_no_break(self):
        """First scan has no previous baseline."""
        current_scan = [
            {'statistic': {'score': 2.5}, 'window': {}, 'n_events': 100}
        ]

        result = detect_break(current_scan, None, threshold=1.5, top_k=10)

        self.assertFalse(result['break_detected'])
        self.assertEqual(result['current_top_score'], 2.5)
        self.assertIsNone(result['previous_top_score'])
        self.assertIn('First scan', result['message'])

    def test_no_break_above_threshold(self):
        """No break when score is above threshold."""
        previous_scan = [
            {'statistic': {'score': 2.5}, 'window': {}, 'n_events': 100}
        ]
        current_scan = [
            {'statistic': {'score': 2.3}, 'window': {}, 'n_events': 100}
        ]

        result = detect_break(current_scan, previous_scan, threshold=1.5, top_k=10)

        self.assertFalse(result['break_detected'])
        self.assertEqual(result['current_top_score'], 2.3)
        self.assertEqual(result['previous_top_score'], 2.5)
        self.assertIn('above threshold', result['message'])

    def test_break_below_threshold(self):
        """Detect break when score drops below threshold."""
        previous_scan = [
            {'statistic': {'score': 2.5}, 'window': {}, 'n_events': 100}
        ]
        current_scan = [
            {'statistic': {'score': 1.2}, 'window': {}, 'n_events': 100}
        ]

        result = detect_break(current_scan, previous_scan, threshold=1.5, top_k=10)

        self.assertTrue(result['break_detected'])
        self.assertEqual(result['current_top_score'], 1.2)
        self.assertEqual(result['previous_top_score'], 2.5)
        self.assertEqual(result['deterioration'], 1.3)
        self.assertIn('below threshold', result['message'])

    def test_break_with_negative_scores(self):
        """Detect break with negative trend scores."""
        previous_scan = [
            {'statistic': {'score': -2.5}, 'window': {}, 'n_events': 100}
        ]
        current_scan = [
            {'statistic': {'score': -2.2}, 'window': {}, 'n_events': 100}
        ]

        result = detect_break(current_scan, previous_scan, threshold=1.5, top_k=10)

        self.assertFalse(result['break_detected'])
        self.assertEqual(result['current_top_score'], 2.2)

    def test_empty_current_scan(self):
        """Handle empty current scan."""
        previous_scan = [
            {'statistic': {'score': 2.5}, 'window': {}, 'n_events': 100}
        ]

        result = detect_break([], previous_scan, threshold=1.5, top_k=10)

        self.assertFalse(result['break_detected'])
        self.assertEqual(result['current_top_score'], 0.0)
        self.assertIn('No valid trends', result['message'])


class TestIntegration(TestCase):
    """Integration tests for full trend scanning workflow."""

    def test_full_workflow_with_structural_break(self):
        """Simulate full workflow detecting a structural break."""
        np.random.seed(42)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2000, freq='1h')
        })

        returns_stable = np.random.normal(0.01, 0.01, 1000)
        returns_break = np.random.normal(0.0, 0.03, 1000)
        returns = np.concatenate([returns_stable, returns_break])

        windows = generate_trend_windows(
            df,
            window_sizes=[250, 500, 1000],
            step=100,
            min_obs=100
        )

        scan_stable = scan_trends(
            returns[:1000],
            weights=None,
            event_end_idx=None,
            windows=[w for w in windows if w['end_idx'] <= 1000],
            embargo_bars=0,
            method='sharpe'
        )

        scan_break = scan_trends(
            returns[1000:],
            weights=None,
            event_end_idx=None,
            windows=[
                {
                    'start_idx': w['start_idx'] - 1000,
                    'end_idx': w['end_idx'] - 1000,
                    'size': w['size'],
                    'timestamp_start': w['timestamp_start'],
                    'timestamp_end': w['timestamp_end']
                }
                for w in windows if w['start_idx'] >= 1000 and w['end_idx'] <= 2000
            ],
            embargo_bars=0,
            method='sharpe'
        )

        top_stable = rank_trends(scan_stable, top_k=5)
        top_break = rank_trends(scan_break, top_k=5)

        if top_stable and top_break:
            stable_score = abs(top_stable[0]['statistic']['score'])
            break_score = abs(top_break[0]['statistic']['score'])

            self.assertGreater(stable_score, break_score)

    def test_purged_evaluation_reduces_events(self):
        """Verify purging reduces event count in windows."""
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0.01, 0.02, n)
        event_end_idx = np.minimum(np.arange(n) + 20, n - 1)

        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1h')
        })
        windows = generate_trend_windows(df, window_sizes=[100], step=100, min_obs=50)

        results_no_purge = scan_trends(returns, None, None, windows, 0, 'sharpe')
        results_purged = scan_trends(returns, None, event_end_idx, windows, 10, 'sharpe')

        for no_purge, purged in zip(results_no_purge, results_purged, strict=False):
            self.assertGreaterEqual(
                no_purge['n_events'],
                purged['n_events']
            )
