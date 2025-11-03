"""Trend Scanning - Detect When Your Strategy Stops Working

This module implements the trend scanning technique from AFML Chapter 15. Instead of
waiting for your P&L to crash before noticing a problem, trend scanning continuously
monitors the strength of trends in your returns and alerts you when they deteriorate.

The core idea: If your strategy works, there should be strong trends (positive Sharpe
ratios) in your recent returns. If those trends suddenly weaken or disappear, it's a
warning sign that market conditions changed and your edge might be gone.

How it works:
1. Define candidate windows: Different time periods to analyze (e.g., last 250 bars,
   last 500 bars, last 1000 bars)
2. Compute trend statistics: For each window, calculate Sharpe ratio or t-statistic
   of returns. Higher = stronger trend.
3. Rank trends: Find which windows show the strongest trends
4. Monitor deterioration: Compare current top trend strength to previous scans. If it
   drops below a threshold, trigger an alert (structural break detected).

Typical usage: Run trend scanning on realized P&L from live positions. If the top
trend Sharpe drops from 2.0 to 0.5, pause trading and investigate what changed.

This is much better than simply watching cumulative P&L, because trend scanning
catches deterioration early (while trends are weakening) rather than late (after
you've already lost money).
"""
import numpy as np
import pandas as pd


def generate_trend_windows(
    df: pd.DataFrame,
    window_sizes: list[int],
    step: int,
    min_obs: int = 30
) -> list[dict]:
    """Generate candidate trend windows for scanning.

    Args:
        df: DataFrame with at least 'timestamp' column
        window_sizes: List of bar-count windows (e.g., [250, 500, 1000])
        step: Bar stride between consecutive candidate windows
        min_obs: Minimum observations per window

    Returns:
        List of window dicts with keys: start_idx, end_idx, size, timestamp_start, timestamp_end
    """
    n = len(df)
    windows = []

    for window_size in window_sizes:
        if window_size < min_obs:
            continue

        start_idx = 0
        while start_idx + window_size <= n:
            end_idx = start_idx + window_size

            window = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': window_size,
                'timestamp_start': df.iloc[start_idx]['timestamp'],
                'timestamp_end': df.iloc[end_idx - 1]['timestamp']
            }
            windows.append(window)

            start_idx += step

    return windows


def compute_trend_statistic(
    returns: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = 'sharpe'
) -> dict:
    """Compute trend statistic (Sharpe or t-statistic) for a window.

    Args:
        returns: Array of returns for the window
        weights: Optional sample weights (concurrency-based)
        method: 'sharpe' or 't_stat'

    Returns:
        Dict with keys: score, mean, std, p_value
    """
    if len(returns) == 0:
        return {
            'score': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'p_value': 1.0
        }

    if weights is None:
        weights = np.ones(len(returns))

    weights = weights / weights.sum()

    weighted_mean = np.sum(returns * weights)
    weighted_var = np.sum(weights * (returns - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_var)

    if weighted_std == 0:
        return {
            'score': 0.0,
            'mean': float(weighted_mean),
            'std': 0.0,
            'p_value': 1.0
        }

    if method == 'sharpe':
        score = weighted_mean / weighted_std
    elif method == 't_stat':
        n_eff = 1.0 / np.sum(weights ** 2)
        score = weighted_mean / (weighted_std / np.sqrt(n_eff))
    else:
        raise ValueError(f"Unknown method: {method}")

    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(score)))

    return {
        'score': float(score),
        'mean': float(weighted_mean),
        'std': float(weighted_std),
        'p_value': float(p_value)
    }


def _apply_purged_window(
    window_start: int,
    window_end: int,
    event_end_idx: np.ndarray,
    embargo_bars: int
) -> np.ndarray:
    """Apply purging to a window: exclude events that overlap outside the window.

    Args:
        window_start: Window start index
        window_end: Window end index (exclusive)
        event_end_idx: Array mapping event index to event end index
        embargo_bars: Right-side embargo in bars

    Returns:
        Boolean mask of valid events within window
    """
    n = len(event_end_idx)
    event_starts = np.arange(n)

    in_window = (event_starts >= window_start) & (event_starts < window_end)
    not_embargoed = event_end_idx < (window_end + embargo_bars)
    mask = in_window & not_embargoed

    return mask


def scan_trends(
    returns: np.ndarray,
    weights: np.ndarray | None,
    event_end_idx: np.ndarray | None,
    windows: list[dict],
    embargo_bars: int = 0,
    method: str = 'sharpe'
) -> list[dict]:
    """Scan trends across multiple candidate windows with purged evaluation.

    Args:
        returns: Full array of returns (one per event/bar)
        weights: Optional sample weights
        event_end_idx: Optional array mapping event index to event end index (for purging)
        windows: List of window dicts from generate_trend_windows
        embargo_bars: Right-side embargo for purging
        method: 'sharpe' or 't_stat'

    Returns:
        List of scan results, each dict with keys:
            window (original window dict), statistic (from compute_trend_statistic),
            n_events (number of events in purged window)
    """
    results = []

    for window in windows:
        start_idx = window['start_idx']
        end_idx = window['end_idx']

        if event_end_idx is not None and embargo_bars > 0:
            mask = _apply_purged_window(start_idx, end_idx, event_end_idx, embargo_bars)
            window_returns = returns[mask]
            window_weights = weights[mask] if weights is not None else None
        else:
            window_returns = returns[start_idx:end_idx]
            window_weights = weights[start_idx:end_idx] if weights is not None else None

        statistic = compute_trend_statistic(window_returns, window_weights, method)

        result = {
            'window': window,
            'statistic': statistic,
            'n_events': len(window_returns)
        }
        results.append(result)

    return results


def rank_trends(scan_results: list[dict], top_k: int = 10) -> list[dict]:
    """Rank trends by score and return top-k.

    Args:
        scan_results: Output from scan_trends
        top_k: Number of top trends to return

    Returns:
        List of top-k scan results sorted by score descending
    """
    sorted_results = sorted(
        scan_results,
        key=lambda x: abs(x['statistic']['score']),
        reverse=True
    )
    return sorted_results[:top_k]


def detect_break(
    current_scan: list[dict],
    previous_scan: list[dict] | None,
    threshold: float,
    top_k: int = 10
) -> dict:
    """Detect structural break by comparing current vs previous trend strength.

    Args:
        current_scan: Latest scan_trends output
        previous_scan: Previous scan_trends output (None if first scan)
        threshold: Break threshold (e.g., 1.5 for Sharpe, 2.0 for t-stat)
        top_k: Number of top trends to monitor

    Returns:
        Dict with keys: break_detected (bool), current_top_score, previous_top_score,
                       deterioration, message
    """
    current_top = rank_trends(current_scan, top_k)

    if not current_top:
        return {
            'break_detected': False,
            'current_top_score': 0.0,
            'previous_top_score': None,
            'deterioration': None,
            'message': 'No valid trends in current scan'
        }

    current_top_score = abs(current_top[0]['statistic']['score'])

    if previous_scan is None:
        return {
            'break_detected': False,
            'current_top_score': current_top_score,
            'previous_top_score': None,
            'deterioration': None,
            'message': 'First scan - no previous baseline'
        }

    previous_top = rank_trends(previous_scan, top_k)
    if not previous_top:
        return {
            'break_detected': False,
            'current_top_score': current_top_score,
            'previous_top_score': None,
            'deterioration': None,
            'message': 'No valid trends in previous scan'
        }

    previous_top_score = abs(previous_top[0]['statistic']['score'])

    if current_top_score < threshold:
        deterioration = previous_top_score - current_top_score if previous_top_score else 0.0
        return {
            'break_detected': True,
            'current_top_score': current_top_score,
            'previous_top_score': previous_top_score,
            'deterioration': deterioration,
            'message': f'Top trend score {current_top_score:.3f} below threshold {threshold}'
        }

    return {
        'break_detected': False,
        'current_top_score': current_top_score,
        'previous_top_score': previous_top_score,
        'deterioration': 0.0,
        'message': f'Top trend score {current_top_score:.3f} above threshold {threshold}'
    }
