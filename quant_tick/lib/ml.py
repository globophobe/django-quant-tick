import logging
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class PurgedKFold(TimeSeriesSplit):
    """Time-series cross-validation with purging and embargo.

    Uses TimeSeriesSplit for forward-only splits (train always before test), then applies:
    1. Purging: Remove training samples whose event ends after test starts
    2. Embargo: Remove training samples within N bars of test boundaries

    TimeSeriesSplit ensures training data is always before test data chronologically,
    preventing the temporal leakage present in standard KFold symmetric splits.

    The purging removes overlapping events from training, and the embargo creates a
    buffer zone to prevent correlation between recent training samples and test samples.

    For interleaved multi-config data, pass timestamp_idx array to use actual timestamps
    for purging/embargo instead of row indices.
    """

    def __init__(self, n_splits: int = 5, embargo_bars: int = 96, **kwargs) -> None:
        """Initialize purged time-series cross-validator."""
        super().__init__(n_splits=n_splits, **kwargs)
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

                train_timestamps = timestamp_idx[train_idx]
                train_event_ends = event_end_idx[train_idx]

                # TimeSeriesSplit ensures train < test, so all training is before test
                # Apply purging: remove training samples whose events overlap with test
                purge_mask = train_event_ends < test_start_ts

                # Apply embargo: remove training samples too close to test start
                if self.embargo_bars > 0:
                    embargo_start = test_start_ts - self.embargo_bars
                    embargo_mask = train_timestamps < embargo_start
                    final_mask = purge_mask & embargo_mask
                else:
                    final_mask = purge_mask

                purged_train_idx = train_idx[final_mask]
            else:
                # Row-index based purging for non-interleaved data
                test_start_idx = test_idx.min()
                train_event_ends = event_end_idx[train_idx]

                # TimeSeriesSplit ensures train < test (all train_idx < test_start_idx)
                # Apply purging: remove training samples whose events overlap with test
                purge_mask = train_event_ends < test_start_idx

                # Apply embargo: remove training samples too close to test start
                if self.embargo_bars > 0:
                    embargo_start = test_start_idx - self.embargo_bars
                    embargo_mask = train_idx < embargo_start
                    final_mask = purge_mask & embargo_mask
                else:
                    final_mask = purge_mask

                purged_train_idx = train_idx[final_mask]

            # Only yield if we have valid training data after purging
            # Don't fall back to unpurged data - that defeats the purpose of purging
            if len(purged_train_idx) > 0:
                yield purged_train_idx, test_idx
            # If purging removed everything, skip this fold entirely


# LPConfig, DEFAULT_WIDTHS, DEFAULT_ASYMMETRIES now imported from quant_core


def compute_first_touch_bars(
    low: np.ndarray,
    high: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first touch time for each entry bar.

    Scans future price path to find first bar where bound is breached.
    Uses high/low for accuracy.

    Args:
        low: Low prices (n,)
        high: High prices (n,)
        lower_bounds: Lower bound per bar (n,)
        upper_bounds: Upper bound per bar (n,)
        max_horizon: Maximum lookahead in bars

    Returns:
        Tuple of (first_touch_lower, first_touch_upper)
        - Shape: (n-1,) each (last bar dropped, no future available)
        - Values: 1-indexed bar offset (1..max_horizon), or max_horizon+1 if censored

    Example:
        Bar 0: close=100, lower_bound=97, upper_bound=103
        Future prices: [102, 98, 104, ...]
        Result: first_touch_lower[0] = 2 (98 <= 97 at bar i+2)
                first_touch_upper[0] = 3 (104 >= 103 at bar i+3)
    """
    n = len(low)
    first_touch_lower = np.full(n - 1, max_horizon + 1, dtype=np.int32)
    first_touch_upper = np.full(n - 1, max_horizon + 1, dtype=np.int32)

    for i in range(n - 1):
        h_end = min(i + max_horizon + 1, n)
        future_lows = low[i + 1 : h_end]
        future_highs = high[i + 1 : h_end]

        lower_breaches = np.where(future_lows <= lower_bounds[i])[0]
        upper_breaches = np.where(future_highs >= upper_bounds[i])[0]

        if len(lower_breaches) > 0:
            first_touch_lower[i] = lower_breaches[0] + 1

        if len(upper_breaches) > 0:
            first_touch_upper[i] = upper_breaches[0] + 1

    return first_touch_lower, first_touch_upper


def generate_labels(
    df: DataFrame,
    widths: list[float],
    asymmetries: list[float],
    max_horizon: int,
) -> DataFrame:
    """Generate survival training labels using vectorized operations.

    Expands data over (bar, config, k) dimensions. For each entry bar and config,
    creates max_horizon rows (one per time step k). Sets hazard_lower(k) = 1
    if first touch occurs at step k.

    Args:
        df: DataFrame with OHLC and base features
        widths: Range widths to test
        asymmetries: Range asymmetries
        max_horizon: Maximum time steps to generate

    Returns:
        DataFrame with shape (n_bars * n_configs * max_horizon, n_features)

        Schema:
        - Metadata: bar_idx, config_id, k, timestamp, entry_price
        - Config: width, asymmetry, lower_bound_pct, upper_bound_pct,
                  range_width, range_asymmetry, dist_to_lower_pct, dist_to_upper_pct
        - Base features: close, volume, realizedVol, rollingSharpe20, etc.
        - Labels: hazard_lower, hazard_upper, event_lower, event_upper

        Row ordering: Interleaved by (bar_idx, k, config_id)
            [bar0_k1_cfg0, bar0_k1_cfg1, ..., bar0_k2_cfg0, ..., bar1_k1_cfg0, ...]

    Invariants:
        - sum(hazard_lower over k) <= 1 for each (bar, config)
        - hazard_lower(k) = 1 implies event_lower = 1
        - event_lower = 0 implies all hazard_lower(k) = 0 (censored)
    """
    if "close" not in df.columns:
        raise ValueError("Missing close column")
    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column")

    close = df["close"].values
    low = df["low"].values if "low" in df.columns else close.copy()
    high = df["high"].values if "high" in df.columns else close.copy()

    if "low" not in df.columns:
        logger.warning(
            "No 'low' column found, using 'close' for lower bound touch detection. "
            "This may underestimate touch probabilities."
        )

    configs = [(w, a) for w in widths for a in asymmetries]
    n_configs = len(configs)
    n_bars = len(df) - 1

    # Pre-compute first touch times for all configs
    # Shape: (n_configs, n_bars) for each of lower/upper
    first_touch_lower_all = np.zeros((n_configs, n_bars), dtype=np.int32)
    first_touch_upper_all = np.zeros((n_configs, n_bars), dtype=np.int32)
    config_params = np.zeros((n_configs, 2), dtype=np.float64)  # (width, asym)

    for cfg_idx, (width, asym) in enumerate(configs):
        lower_pct = -width * (0.5 - asym)
        upper_pct = width * (0.5 + asym)
        config_params[cfg_idx] = [lower_pct, upper_pct]

        lower_bounds = close * (1 + lower_pct)
        upper_bounds = close * (1 + upper_pct)

        first_touch_lower, first_touch_upper = compute_first_touch_bars(
            low, high, lower_bounds, upper_bounds, max_horizon
        )
        first_touch_lower_all[cfg_idx] = first_touch_lower
        first_touch_upper_all[cfg_idx] = first_touch_upper

    # Create coordinate grids - interleaved by (bar_idx, k, config_id)
    # Pattern: [bar0_k1_cfg0, bar0_k1_cfg1, ..., bar0_k2_cfg0, ...]
    bar_idx_grid = np.repeat(np.arange(n_bars), n_configs * max_horizon)
    k_grid = np.tile(np.repeat(np.arange(1, max_horizon + 1), n_configs), n_bars)
    config_id_grid = np.tile(np.arange(n_configs), n_bars * max_horizon)

    # Build result dictionary
    result_dict = {}

    # Add coordinate columns
    result_dict["bar_idx"] = bar_idx_grid
    result_dict["config_id"] = config_id_grid
    result_dict["k"] = k_grid

    # Broadcast base features from df (excluding last bar)
    df_truncated = df.iloc[:n_bars]
    for col in df_truncated.columns:
        if col in ["close", "timestamp"]:
            # These get special handling
            continue
        result_dict[col] = df_truncated[col].values[bar_idx_grid]

    # Add entry_price (close at entry bar)
    result_dict["entry_price"] = close[bar_idx_grid]
    result_dict["timestamp"] = df_truncated["timestamp"].values[bar_idx_grid]

    # Broadcast config-specific features
    # Extract width and asymmetry for each row based on config_id
    widths_array = np.array([w for w, a in configs])
    asyms_array = np.array([a for w, a in configs])

    result_dict["width"] = widths_array[config_id_grid]
    result_dict["asymmetry"] = asyms_array[config_id_grid]

    # Compute bound percentages
    lower_pct_grid = config_params[config_id_grid, 0]
    upper_pct_grid = config_params[config_id_grid, 1]

    result_dict["lower_bound_pct"] = lower_pct_grid
    result_dict["upper_bound_pct"] = upper_pct_grid
    result_dict["range_width"] = upper_pct_grid - lower_pct_grid
    result_dict["range_asymmetry"] = upper_pct_grid + lower_pct_grid

    # Compute distance features
    close_grid = close[bar_idx_grid]
    lower_bounds_grid = close_grid * (1 + lower_pct_grid)
    upper_bounds_grid = close_grid * (1 + upper_pct_grid)

    result_dict["dist_to_lower_pct"] = (close_grid - lower_bounds_grid) / close_grid
    result_dict["dist_to_upper_pct"] = (upper_bounds_grid - close_grid) / close_grid

    # Extract first touch times using advanced indexing
    # first_touch_lower_all[config_id, bar_idx]
    t_lower_grid = first_touch_lower_all[config_id_grid, bar_idx_grid]
    t_upper_grid = first_touch_upper_all[config_id_grid, bar_idx_grid]

    # Compute hazard and event labels
    result_dict["hazard_lower"] = (t_lower_grid == k_grid).astype(np.int8)
    result_dict["hazard_upper"] = (t_upper_grid == k_grid).astype(np.int8)
    result_dict["event_lower"] = (t_lower_grid <= max_horizon).astype(np.int8)
    result_dict["event_upper"] = (t_upper_grid <= max_horizon).astype(np.int8)

    # Create DataFrame
    result = pd.DataFrame(result_dict)

    logger.info(
        f"Generated survival labels: {len(result)} rows = "
        f"{n_bars} bars × {n_configs} configs × {max_horizon} time steps"
    )

    return result






def calibrate_per_horizon(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = "auto",
) -> tuple[IsotonicRegression | LogisticRegression | None, str]:
    """Fit calibrator on validation set with auto-selection and validation.

    Args:
        y_true: True binary labels (0/1)
        y_pred_proba: Predicted probabilities
        method: 'auto', 'isotonic', or 'platt'
            'auto': Use isotonic if >=50 samples, else Platt

    Returns:
        Tuple of (calibrator, method_used)
        Returns (None, "none") if calibration fails or makes predictions worse
    """
    from sklearn.metrics import brier_score_loss

    n_samples = len(y_true)

    # Check for degenerate cases
    if n_samples < 10:
        logger.warning(f"Too few samples for calibration ({n_samples}), skipping")
        return None, "none"

    # Check for constant predictions
    unique_preds = np.unique(y_pred_proba)
    if len(unique_preds) < 3:
        logger.warning(
            f"Predictions lack variance ({len(unique_preds)} unique), skipping calibration"
        )
        return None, "none"

    # Check for class imbalance (need at least a few positive samples)
    n_positive = y_true.sum()
    if n_positive < 3 or n_positive > (n_samples - 3):
        logger.warning(
            f"Extreme class imbalance ({n_positive}/{n_samples}), skipping calibration"
        )
        return None, "none"

    # Compute baseline Brier score
    brier_before = brier_score_loss(y_true, y_pred_proba)

    if method == "auto":
        method = "isotonic" if n_samples >= 50 else "platt"

    try:
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_pred_proba, y_true)
            y_calibrated = calibrator.transform(y_pred_proba)
        elif method == "platt":
            lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            lr.fit(y_pred_proba.reshape(-1, 1), y_true)
            calibrator = lr
            y_calibrated = lr.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
        else:
            return None, "none"

        # Validate calibration improves or maintains performance
        brier_after = brier_score_loss(y_true, y_calibrated)
        if brier_after > brier_before * 1.05:  # Allow 5% tolerance for noise
            logger.warning(
                f"Calibration worsened Brier score ({brier_before:.4f} -> {brier_after:.4f}), skipping"
            )
            return None, "none"

        logger.debug(
            f"Calibration ({method}): Brier {brier_before:.4f} -> {brier_after:.4f}"
        )
        return calibrator, method

    except Exception as e:
        logger.warning(f"Calibration failed: {e}, skipping")
        return None, "none"








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


