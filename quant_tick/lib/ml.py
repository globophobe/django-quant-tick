from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold


class PurgedKFold(KFold):
    """Time-series cross-validation that prevents label leakage from overlapping events.

    Standard KFold fails for financial time series because events overlap: a trade opened
    at time T might not close until T+horizon, overlapping with the test period. If we
    train on this sample, we're using information from the future (label leakage).

    PurgedKFold fixes this two ways:
    1. Purging: Remove training samples whose event ends after the test period starts
    2. Embargo: Remove training samples within N bars after test end (prevents correlation
       between recent training samples and test samples)

    The embargo is critical: even after purging, recent training samples just before the
    test set may be correlated with test samples. The embargo creates a buffer zone.

    For interleaved multi-config data, pass timestamp_idx array to use actual timestamps
    for purging/embargo instead of row indices.
    """

    def __init__(self, n_splits: int = 5, embargo_bars: int = 96, **kwargs) -> None:
        """Initialize purged k-fold cross-validator."""
        super().__init__(n_splits=n_splits, shuffle=False, **kwargs)
        self.embargo_bars = embargo_bars

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        event_end_idx: np.ndarray | None = None,
        timestamp_idx: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test splits.

        Args:
            X: Feature matrix
            y: Labels (unused)
            groups: Groups (unused)
            event_end_idx: Array of event end timestamps (timestamp_idx + horizon)
            timestamp_idx: Array of timestamp indices for each row. If provided,
                purging/embargo use timestamp space instead of row index space.
        """
        if event_end_idx is None:
            yield from super().split(X, y, groups)
            return

        for train_idx, test_idx in super().split(X, y, groups):
            # Get timestamp boundaries for test set
            if timestamp_idx is not None:
                test_timestamps = timestamp_idx[test_idx]
                test_start_ts = test_timestamps.min()
                test_end_ts = test_timestamps.max()

                train_timestamps = timestamp_idx[train_idx]

                # Split by timestamp, not row index
                before_mask = train_timestamps < test_start_ts
                after_mask = train_timestamps > test_end_ts

                before_train = train_idx[before_mask]
                after_train = train_idx[after_mask]

                # Purge: remove samples BEFORE test whose events overlap test
                if len(before_train) > 0:
                    before_event_ends = event_end_idx[before_train]
                    purge_mask = before_event_ends < test_start_ts
                    purged_before = before_train[purge_mask]
                else:
                    purged_before = before_train

                # Embargo: remove samples AFTER test that are too close
                if self.embargo_bars > 0 and len(after_train) > 0:
                    after_timestamps = timestamp_idx[after_train]
                    embargo_end_ts = test_end_ts + self.embargo_bars
                    embargo_mask = after_timestamps > embargo_end_ts
                    purged_after = after_train[embargo_mask]
                else:
                    purged_after = after_train
            else:
                # Original logic for non-interleaved data
                test_start_idx = test_idx.min()
                test_end_idx = test_idx.max()

                before_test = train_idx[train_idx < test_start_idx]
                after_test = train_idx[train_idx > test_end_idx]

                if len(before_test) > 0:
                    before_event_ends = event_end_idx[before_test]
                    purge_mask = before_event_ends < test_start_idx
                    purged_before = before_test[purge_mask]
                else:
                    purged_before = before_test

                if self.embargo_bars > 0 and len(after_test) > 0:
                    embargo_end = test_end_idx + self.embargo_bars
                    embargo_mask = after_test > embargo_end
                    purged_after = after_test[embargo_mask]
                else:
                    purged_after = after_test

            purged_train_idx = np.concatenate([purged_before, purged_after])

            if len(purged_train_idx) == 0:
                purged_train_idx = train_idx

            yield purged_train_idx, test_idx


@dataclass
class LPConfig:
    """Optimal LP configuration."""

    lower_pct: float
    upper_pct: float
    borrow_ratio: float
    p_touch_lower: float
    p_touch_upper: float
    width: float
    asymmetry: float


# Default bounds grid for training and inference
DEFAULT_WIDTHS = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
DEFAULT_ASYMMETRIES = [-0.4, -0.2, 0.0, 0.2, 0.4]


def generate_touch_labels(
    df: DataFrame,
    lower_pct: float,
    upper_pct: float,
    horizon_bars: int,
) -> DataFrame:
    """Generate touch labels for given bounds.

    Args:
        df: DataFrame with 'close' column (and optionally 'timestamp')
        lower_pct: Lower bound as fraction (e.g., -0.03 for -3%)
        upper_pct: Upper bound as fraction (e.g., 0.05 for +5%)
        horizon_bars: Number of bars to look ahead

    Returns:
        DataFrame with added columns:
        - touched_lower: 1 if price touched lower bound within horizon, else 0
        - touched_upper: 1 if price touched upper bound within horizon, else 0
        - entry_price: Price at entry (close)
        - lower_bound: Actual lower bound price
        - upper_bound: Actual upper bound price
    """
    close = df["close"].values
    n = len(close)

    touched_lower = np.zeros(n, dtype=np.int8)
    touched_upper = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        entry_price = close[i]
        lower_bound = entry_price * (1 + lower_pct)
        upper_bound = entry_price * (1 + upper_pct)

        end_idx = min(i + horizon_bars, n)
        future_prices = close[i + 1 : end_idx]

        if len(future_prices) > 0:
            if np.any(future_prices <= lower_bound):
                touched_lower[i] = 1
            if np.any(future_prices >= upper_bound):
                touched_upper[i] = 1

    result = df.copy()
    result["touched_lower"] = touched_lower
    result["touched_upper"] = touched_upper
    result["entry_price"] = close
    result["lower_bound_pct"] = lower_pct
    result["upper_bound_pct"] = upper_pct
    result["horizon_bars"] = horizon_bars

    return result


def generate_multi_config_labels(
    df: DataFrame,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
    horizon_bars: int = 60,
) -> DataFrame:
    """Generate touch labels for multiple bound configurations.

    Creates an augmented dataset where each row is replicated for each config,
    allowing the model to learn how features relate to touch probability
    at different bound widths/asymmetries.

    IMPORTANT: Rows are interleaved by timestamp (all configs for T0, then T1, etc.)
    to ensure PurgedKFold splits respect time ordering. The `timestamp_idx` column
    tracks the original bar index for proper event_end calculation in CV.

    Args:
        df: DataFrame with features and 'close' column
        widths: List of range widths (e.g., [0.03, 0.05, 0.07])
        asymmetries: List of asymmetries (e.g., [-0.2, 0, 0.2])
        horizon_bars: Prediction horizon

    Returns:
        Augmented DataFrame with config features, touch labels, and timestamp_idx
    """
    if widths is None:
        widths = DEFAULT_WIDTHS
    if asymmetries is None:
        asymmetries = DEFAULT_ASYMMETRIES

    n_bars = len(df)
    configs = [(w, a) for w in widths for a in asymmetries]
    n_configs = len(configs)

    # Pre-generate labels for all configs
    config_labels = {}
    for width, asym in configs:
        lower_pct = -width * (0.5 - asym)
        upper_pct = width * (0.5 + asym)
        labeled = generate_touch_labels(df, lower_pct, upper_pct, horizon_bars)
        labeled["width"] = width
        labeled["asymmetry"] = asym
        config_labels[(width, asym)] = labeled

    # Interleave by timestamp: all configs for bar 0, then bar 1, etc.
    rows = []
    for bar_idx in range(n_bars):
        for width, asym in configs:
            row = config_labels[(width, asym)].iloc[[bar_idx]].copy()
            row["timestamp_idx"] = bar_idx
            rows.append(row)

    result = pd.concat(rows, ignore_index=True)

    # Verify interleaving: row i should have timestamp_idx = i // n_configs
    assert all(result["timestamp_idx"] == np.arange(len(result)) // n_configs)

    return result


def compute_bound_features(
    df: DataFrame,
    lower_pct: float,
    upper_pct: float,
) -> DataFrame:
    """Add bound-related features to dataframe.

    Args:
        df: DataFrame with 'close' column
        lower_pct: Lower bound percentage
        upper_pct: Upper bound percentage

    Returns:
        DataFrame with added bound features
    """
    result = df.copy()

    result["lower_bound_pct"] = lower_pct
    result["upper_bound_pct"] = upper_pct
    result["range_width"] = upper_pct - lower_pct
    result["range_asymmetry"] = upper_pct + lower_pct

    # Distance from current price to bounds (normalized)
    if "close" in df.columns:
        close = df["close"].values
        entry = close  # Current close as entry price
        lower_bound = entry * (1 + lower_pct)
        upper_bound = entry * (1 + upper_pct)

        result["dist_to_lower_pct"] = (close - lower_bound) / close
        result["dist_to_upper_pct"] = (upper_bound - close) / close

    return result


def find_optimal_config(
    predict_lower_fn: callable,
    predict_upper_fn: callable,
    features: DataFrame,
    touch_tolerance: float = 0.15,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
) -> LPConfig | None:
    """Find the tightest LP range where both touch probabilities are below tolerance.

    Args:
        predict_lower_fn: Function(features, lower_pct, upper_pct) -> P(touch lower)
        predict_upper_fn: Function(features, lower_pct, upper_pct) -> P(touch upper)
        features: Feature DataFrame for current bar
        touch_tolerance: Maximum acceptable touch probability
        widths: Range widths to search (tightest first)
        asymmetries: Asymmetries to search

    Returns:
        LPConfig with optimal bounds and borrow_ratio, or None if no valid config
    """
    if widths is None:
        widths = sorted(DEFAULT_WIDTHS)  # Tightest first
    if asymmetries is None:
        asymmetries = DEFAULT_ASYMMETRIES

    for width in widths:
        valid_configs = []

        for asym in asymmetries:
            lower_pct = -width * (0.5 - asym)
            upper_pct = width * (0.5 + asym)

            p_lower = predict_lower_fn(features, lower_pct, upper_pct)
            p_upper = predict_upper_fn(features, lower_pct, upper_pct)

            if p_lower < touch_tolerance and p_upper < touch_tolerance:
                # Compute skew: positive = price more likely to go up
                skew = p_upper - p_lower

                valid_configs.append(
                    {
                        "lower_pct": lower_pct,
                        "upper_pct": upper_pct,
                        "p_lower": p_lower,
                        "p_upper": p_upper,
                        "skew": skew,
                        "asym": asym,
                        "width": width,
                    }
                )

        if valid_configs:
            # Pick best config based on skew magnitude
            # When |skew| is small, prefer neutral (asym=0) and lowest max prob
            # When |skew| is large, align asymmetry with skew direction
            skew_threshold = 0.05

            def config_score(c: dict, threshold: float = skew_threshold) -> tuple:
                """Score config: (skew_alignment, -max_prob, neutral_preference)."""
                abs_skew = abs(c["skew"])
                max_prob = max(c["p_lower"], c["p_upper"])

                if abs_skew < threshold:
                    # Low skew: prefer neutral (asym=0) and lowest max prob
                    return (0, -abs(c["asym"]), -max_prob)
                else:
                    # High skew: prefer aligned asymmetry
                    alignment = c["skew"] * c["asym"]
                    return (1, alignment, -max_prob)

            best = max(valid_configs, key=config_score)

            # Borrow ratio: 0.5 + asymmetry
            # Higher asymmetry (bullish) -> borrow more USDC -> higher borrow_ratio
            borrow_ratio = 0.5 + best["asym"]

            return LPConfig(
                lower_pct=best["lower_pct"],
                upper_pct=best["upper_pct"],
                borrow_ratio=borrow_ratio,
                p_touch_lower=best["p_lower"],
                p_touch_upper=best["p_upper"],
                width=best["width"],
                asymmetry=best["asym"],
            )

    return None


def check_position_change_allowed(
    bars_since_last_change: int,
    min_hold_bars: int = 15,
) -> bool:
    """Check if position change is allowed based on constraints.

    Args:
        bars_since_last_change: Bars since last position/config change
        min_hold_bars: Minimum bars before allowing change

    Returns:
        True if change is allowed
    """
    return bars_since_last_change >= min_hold_bars
