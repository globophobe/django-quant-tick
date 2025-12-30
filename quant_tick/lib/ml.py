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
    1. Purging: Remove training samples whose event ends after test starts (prevents label leakage)
    2. Embargo: Remove training samples within N bars BEFORE test starts (creates buffer zone)

    TimeSeriesSplit ensures training data is always before test data chronologically,
    preventing the temporal leakage present in standard KFold symmetric splits.

    The purging removes overlapping events from training. The embargo creates a
    temporal gap [test_start - embargo_bars, test_start) to prevent overfitting
    from samples in temporal proximity to the test period.

    For interleaved multi-config data, pass bar_idx array to use bar indices
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
        event_end_exclusive_idx: np.ndarray | None = None,
        bar_idx: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test splits.

        Args:
            X: Feature matrix
            y: Labels (unused)
            groups: Groups (unused)
            event_end_exclusive_idx: Array of exclusive upper bounds (bar_idx + horizon + 1)
            bar_idx: Array of integer bar indices (0, 1, 2, ...) for each row. If provided,
                purging/embargo use bar index space instead of row index space.
        """
        if event_end_exclusive_idx is None:
            yield from super().split(X, y, groups)
            return

        for train_idx, test_idx in super().split(X, y, groups):
            # Get bar index boundaries for test set
            if bar_idx is not None:
                test_bars = bar_idx[test_idx]
                test_start_bar = test_bars.min()

                train_bars = bar_idx[train_idx]
                train_event_ends = event_end_exclusive_idx[train_idx]

                # TimeSeriesSplit ensures train < test, so all training is before test
                # Apply purging: remove training samples whose events overlap with test
                # For exclusive end, keep samples where event_end_exclusive <= test_start (no overlap)
                purge_mask = train_event_ends <= test_start_bar

                # Apply embargo: remove training samples too close to test start
                if self.embargo_bars > 0:
                    embargo_start = test_start_bar - self.embargo_bars
                    embargo_mask = train_bars < embargo_start
                    final_mask = purge_mask & embargo_mask
                else:
                    final_mask = purge_mask

                purged_train_idx = train_idx[final_mask]
            else:
                # Row-index based purging for non-interleaved data
                test_start_idx = test_idx.min()
                train_event_ends = event_end_exclusive_idx[train_idx]

                # TimeSeriesSplit ensures train < test (all train_idx < test_start_idx)
                # Apply purging: remove training samples whose events overlap with test
                # For exclusive end, keep samples where event_end_exclusive <= test_start (no overlap)
                purge_mask = train_event_ends <= test_start_idx

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
    horizons: list[int],
) -> DataFrame:
    """Generate multi-horizon competing-risks labels.

    Creates dataset with bars × configs rows (NOT expanded by horizon dimension).
    Adds multiclass label columns for each horizon indicating which barrier
    gets hit first: UP_FIRST (0), DOWN_FIRST (1), or TIMEOUT (2).

    Args:
        df: DataFrame with OHLC and base features
        widths: Range widths to test
        asymmetries: Range asymmetries
        horizons: Prediction horizons in bars (e.g., [48, 96, 144, ...])

    Returns:
        DataFrame with shape (n_bars * n_configs, n_features)

        Schema:
        - Metadata: bar_idx, config_id
        - Config: width, asymmetry, lower_bound_pct, upper_bound_pct,
                  range_width, range_asymmetry, dist_to_lower_pct, dist_to_upper_pct
        - Base features: close, volume, realizedVol, rollingSharpe20, etc.
        - Labels: first_hit_h{H} for each H in horizons
            Values: 0=UP_FIRST, 1=DOWN_FIRST, 2=TIMEOUT

    Memory: ~180x reduction vs hazard models
        (bars × configs vs bars × configs × max_horizon)
    """
    if "close" not in df.columns:
        raise ValueError("Missing close column")

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
    n_bars = len(df) - 1  # Drop last bar (no future for labels)

    # Compute first touch for each config using existing logic
    max_horizon = max(horizons)
    first_touch_lower_all = np.zeros((n_configs, n_bars), dtype=np.int32)
    first_touch_upper_all = np.zeros((n_configs, n_bars), dtype=np.int32)

    for cfg_idx, (width, asym) in enumerate(configs):
        lower_pct = -width * (0.5 - asym)
        upper_pct = width * (0.5 + asym)
        lower_bounds = close * (1 + lower_pct)
        upper_bounds = close * (1 + upper_pct)

        first_touch_lower, first_touch_upper = compute_first_touch_bars(
            low, high, lower_bounds, upper_bounds, max_horizon
        )
        first_touch_lower_all[cfg_idx] = first_touch_lower
        first_touch_upper_all[cfg_idx] = first_touch_upper

    # Expand to bars × configs (WITHOUT object columns like timestamp)
    bar_idx_grid = np.repeat(np.arange(n_bars), n_configs)
    config_id_grid = np.tile(np.arange(n_configs), n_bars)

    result_dict = {"bar_idx": bar_idx_grid, "config_id": config_id_grid}

    # Only include numeric and bool columns, explicitly exclude metadata and labels
    base_feature_cols = [
        c for c in df.select_dtypes(include=["number", "bool"]).columns
        if c not in ["bar_idx", "config_id", "k", "hazard_lower", "hazard_upper"]
        and not c.startswith("touched_")
    ]

    for col in base_feature_cols:
        if col in df.columns:
            result_dict[col] = df[col].values[:n_bars][bar_idx_grid]

    # Preserve timestamp for backtest position tracking
    if "timestamp" in df.columns:
        result_dict["timestamp"] = df["timestamp"].values[:n_bars][bar_idx_grid]

    # Add config features
    widths_array = np.array([w for w, a in configs])
    asyms_array = np.array([a for w, a in configs])
    result_dict["width"] = widths_array[config_id_grid]
    result_dict["asymmetry"] = asyms_array[config_id_grid]

    # Compute bound percentages
    lower_pcts = np.array([-w * (0.5 - a) for w, a in configs])
    upper_pcts = np.array([w * (0.5 + a) for w, a in configs])

    result_dict["lower_bound_pct"] = lower_pcts[config_id_grid]
    result_dict["upper_bound_pct"] = upper_pcts[config_id_grid]
    result_dict["range_width"] = (upper_pcts - lower_pcts)[config_id_grid]
    result_dict["range_asymmetry"] = (upper_pcts + lower_pcts)[config_id_grid]

    # Compute distance features
    close_grid = close[:n_bars][bar_idx_grid]
    lower_bounds_grid = close_grid * (1 + lower_pcts[config_id_grid])
    upper_bounds_grid = close_grid * (1 + upper_pcts[config_id_grid])

    result_dict["dist_to_lower_pct"] = (close_grid - lower_bounds_grid) / close_grid
    result_dict["dist_to_upper_pct"] = (upper_bounds_grid - close_grid) / close_grid

    # Compute per-row touch times (vectorized, no loop)
    touch_lower_grid = first_touch_lower_all[config_id_grid, bar_idx_grid]
    touch_upper_grid = first_touch_upper_all[config_id_grid, bar_idx_grid]

    # Generate competing-risks labels (multiclass per horizon)
    for H in horizons:
        # Vectorized comparison (no Python loop)
        labels = np.full(n_bars * n_configs, 2, dtype=np.int8)  # Default: TIMEOUT

        # Ties (touch_upper == touch_lower) go to UP_FIRST (deterministic)
        up_first_mask = (touch_upper_grid <= touch_lower_grid) & (touch_upper_grid <= H)
        down_first_mask = (touch_lower_grid < touch_upper_grid) & (touch_lower_grid <= H)

        labels[up_first_mask] = 0  # UP_FIRST
        labels[down_first_mask] = 1  # DOWN_FIRST

        result_dict[f"first_hit_h{H}"] = labels

    result = pd.DataFrame(result_dict)

    logger.info(
        f"Generated multi-horizon labels: {len(result)} rows = "
        f"{n_bars} bars × {n_configs} configs (not expanded by {len(horizons)} horizons)"
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


def calibrate_multiclass_ovr(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = "auto",
) -> tuple[dict | None, dict | None]:
    """Calibrate multiclass probabilities using One-vs-Rest.

    Args:
        y_true: True class labels (0, 1, 2)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        method: 'auto', 'isotonic', or 'platt'

    Returns:
        (calibrators_dict, methods_dict) where:
        - calibrators_dict: {0: calibrator, 1: calibrator, 2: None if skipped, ...}
        - methods_dict: {0: "isotonic", 1: "platt", 2: "none", ...}
        Returns (None, None) if entire calibration fails

    Note: Individual classes can skip calibration (stored as None with method="none")
    while others calibrate successfully.
    """
    from sklearn.metrics import log_loss

    n_samples, n_classes = y_pred_proba.shape

    # Validate minimum samples
    if n_samples < 30:
        logger.warning(f"Too few samples for multiclass calibration ({n_samples}), skipping")
        return None, None

    # Validate class distribution
    for cls in range(n_classes):
        if (y_true == cls).sum() < 5:
            logger.warning(f"Too few samples for class {cls}, skipping all calibration")
            return None, None

    # Baseline log-loss
    logloss_before = log_loss(y_true, y_pred_proba)

    # Train one binary calibrator per class (some may fail)
    calibrators = {}
    methods = {}
    for cls in range(n_classes):
        y_binary = (y_true == cls).astype(int)
        y_pred_cls = y_pred_proba[:, cls]

        calibrator, calib_method = calibrate_per_horizon(y_binary, y_pred_cls, method=method)

        calibrators[cls] = calibrator  # Can be None
        methods[cls] = calib_method    # Can be "none"

    # Check if at least one class was calibrated
    if all(m == "none" for m in methods.values()):
        logger.warning("All classes failed calibration, skipping multiclass calibration")
        return None, None

    # CRITICAL: Validate multiclass calibration actually improves log-loss
    # Apply calibration and check if it helps
    from quant_core.prediction import apply_calibration

    calibrated_proba = np.zeros_like(y_pred_proba)
    for cls in range(n_classes):
        if calibrators[cls] is not None and methods[cls] != "none":
            calibrated_proba[:, cls] = apply_calibration(
                y_pred_proba[:, cls], calibrators[cls], methods[cls]
            )
        else:
            calibrated_proba[:, cls] = y_pred_proba[:, cls]

    # Re-normalize
    row_sums = calibrated_proba.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    calibrated_proba = calibrated_proba / row_sums

    # Check if calibration improved multiclass log-loss
    logloss_after = log_loss(y_true, calibrated_proba)

    if logloss_after > logloss_before * 1.05:  # Allow 5% tolerance
        logger.warning(
            f"Multiclass calibration worsened log-loss "
            f"({logloss_before:.4f} -> {logloss_after:.4f}), skipping"
        )
        return None, None

    logger.debug(
        f"Multiclass calibration: methods={methods}, "
        f"log-loss {logloss_before:.4f} -> {logloss_after:.4f}"
    )

    return calibrators, methods






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


