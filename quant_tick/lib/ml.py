import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

MISSING_SENTINEL = -999.0


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
    total_rows = n_bars * n_configs * max_horizon

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


def hazard_to_per_horizon_probs(
    hazard_model: Any,
    X_base: DataFrame,
    feature_cols: list[str],
    decision_horizons: list[int],
    max_horizon: int,
) -> dict[int, float]:
    """Reconstruct per-horizon probabilities from hazard model.

    Predicts h(k) for k=1..max_horizon, composes into survival curve S(k),
    then extracts P(hit_by_H) for decision horizons.

    This bridges the gap between survival models and the existing inference
    interface which expects P(hit_by_60), P(hit_by_120), P(hit_by_180).

    Args:
        hazard_model: Trained LightGBM with calibrator_ attribute
        X_base: Single-row DataFrame with base features (no k column)
        feature_cols: Training feature order from artifact (includes k)
        decision_horizons: Horizons to extract (e.g., [60, 120, 180])
        max_horizon: Maximum k to predict (e.g., 180)

    Returns:
        Dict mapping H -> P(hit_by_H)

        Guarantees:
        - P(hit_by_H1) <= P(hit_by_H2) <= P(hit_by_H3) (monotonic)
        - P(hit_by_H) ∈ [0, 1] (valid probability)

    Example:
        X_base = pd.DataFrame([{"close": 100, "volume": 1000, ...}])
        probs = hazard_to_per_horizon_probs(
            model, X_base, feature_cols, [60, 120, 180], max_horizon=180
        )
        # probs = {60: 0.23, 120: 0.35, 180: 0.42}
    """
    # feature_cols = training feature order from artifact (includes 'k')
    base_row = X_base.iloc[0].to_dict()

    # Expand over k=1..max_horizon
    rows = []
    for k in range(1, max_horizon + 1):
        row = base_row.copy()
        row["k"] = k
        rows.append(row)

    X_expanded = pd.DataFrame(rows)

    # Ensure all training features exist; fill missing with sentinel
    for col in feature_cols:
        if col not in X_expanded.columns:
            X_expanded[col] = MISSING_SENTINEL

    # Reindex columns to training order (critical for LightGBM position-based matching)
    X_expanded = X_expanded[feature_cols]

    # Predict hazard h(k) for k=1..max_horizon
    h_k_raw = hazard_model.predict_proba(X_expanded)[:, 1]

    # Apply calibration (vectorized)
    calibrator = getattr(hazard_model, "calibrator_", None)
    calibration_method = getattr(hazard_model, "calibration_method_", "none")

    h_k = apply_calibration(h_k_raw, calibrator, calibration_method)

    # Clip to [0, 1] for numerical stability
    h_k = np.clip(h_k, 0.0, 1.0)

    # Compute survival curve: S(k) = ∏(j=1 to k) [1 - h(j)]
    S_k = np.cumprod(1.0 - h_k)

    # Convert to cumulative touch: P(hit_by_k) = 1 - S(k)
    P_hit_by_k = 1.0 - S_k

    # Extract decision horizons
    result = {}
    for h in decision_horizons:
        if h <= max_horizon:
            result[h] = float(P_hit_by_k[h - 1])  # k is 1-indexed, array is 0-indexed
        else:
            # Horizon beyond max_horizon: use final value
            result[h] = float(P_hit_by_k[-1])

    # Enforce monotonicity to guarantee contract
    result = enforce_monotonicity(result, log_violations=False)

    return result


def enforce_monotonicity(
    horizon_probs: dict[int, float],
    log_violations: bool = True,
) -> dict[int, float]:
    """Enforce P(hit_by_H) non-decreasing across horizons via cumulative max.

    Args:
        horizon_probs: Dict mapping horizon (int) -> touch probability (float)
        log_violations: If True, log when monotonicity is violated

    Returns:
        Dict with same keys but monotone probabilities (cumulative max)
    """
    if not horizon_probs:
        return {}

    # Sort by horizon
    sorted_horizons = sorted(horizon_probs.keys())

    # Check for violations BEFORE fixing
    violations = []
    for i in range(1, len(sorted_horizons)):
        h_prev = sorted_horizons[i - 1]
        h_curr = sorted_horizons[i]

        if horizon_probs[h_curr] < horizon_probs[h_prev]:
            decrease = horizon_probs[h_prev] - horizon_probs[h_curr]
            violations.append(
                {
                    "from_horizon": h_prev,
                    "to_horizon": h_curr,
                    "decrease": decrease,
                    "prev_prob": horizon_probs[h_prev],
                    "curr_prob": horizon_probs[h_curr],
                }
            )

    if log_violations and violations:
        logger.warning(
            f"Monotonicity violations detected: {len(violations)} decreases. "
            f"Largest: {max(v['decrease'] for v in violations):.4f}"
        )

    # Apply cumulative maximum
    monotone_probs = {}
    cumulative_max = 0.0

    for h in sorted_horizons:
        cumulative_max = max(cumulative_max, horizon_probs[h])
        monotone_probs[h] = cumulative_max

    return monotone_probs


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


def apply_calibration(
    proba: float | np.ndarray,
    calibrator: IsotonicRegression | LogisticRegression | None,
    calibration_method: str,
) -> float | np.ndarray:
    """Apply calibrator to raw probability.

    Supports both scalar and array inputs for vectorized calibration.

    Args:
        proba: Raw predicted probability (scalar or array)
        calibrator: Trained calibrator (isotonic or platt)
        calibration_method: "isotonic", "platt", or "none"

    Returns:
        Calibrated probability (or raw if no calibrator)
        Returns same type as input (scalar -> scalar, array -> array)
    """
    if calibrator is None or calibration_method == "none":
        return proba

    # Track if input was scalar
    is_scalar = np.isscalar(proba)
    proba_array = np.asarray(proba)

    try:
        if calibration_method == "isotonic":
            calibrated = calibrator.transform(proba_array)
        elif calibration_method == "platt":
            calibrated = calibrator.predict_proba(proba_array.reshape(-1, 1))[:, 1]
        else:
            logger.warning(f"Unknown calibration method: {calibration_method}")
            return proba

        # Return scalar if input was scalar
        if is_scalar:
            return float(calibrated[0] if calibrated.ndim > 0 else calibrated)
        return calibrated

    except Exception as e:
        logger.warning(f"Calibration application failed: {e}, using raw probability")
        return proba


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
    """Find tightest range that passes touch probability filter.

    Risk Filtering Logic:
    For each candidate range (defined by width + asymmetry):
    1. Predict P(lower touch) and P(upper touch) using trained models
    2. Apply calibration, isotonic or Platt
    3. Enforce monotonicity across horizons
    4. Check if either probability exceeds touch_tolerance
    5. Reject range if too risky; accept if both probabilities are low enough

    Selection Strategy:
    - Iterate from tightest to widest ranges
    - Return first width that has any valid asymmetry config
    - Among valid asymmetries at that width, pick the one with best combined touch prob
    - Ensures selected range isn't too tight for current volatility

    Args:
        predict_lower_fn: Function(features, horizon) -> P(lower touch)
        predict_upper_fn: Function(features, horizon) -> P(upper touch)
        features: Feature DataFrame for current bar
        touch_tolerance: Max acceptable touch probability (e.g., 0.15 = 15% max risk)
        widths: Range widths to test (sorted tightest to widest)
        asymmetries: Asymmetry values to test (0 = symmetric, +/- = skewed)

    Returns:
        Config with selected width/asymmetry, or None if all ranges too risky
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


def prepare_features(
    df: DataFrame,
    feature_cols: list[str],
    sentinel: float = MISSING_SENTINEL,
) -> tuple[np.ndarray, list[str]]:
    """Prepare features.

    Adds boolean missing indicator columns for each feature with NaN values,
    then fills NaN with sentinel value. This allows the model to distinguish
    "missing" from real zeros.

    If feature_cols already contains *_missing columns (inference mode with frozen schema),
    skips creating new missing columns to prevent schema drift.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        sentinel: Value to fill NaN with

    Returns:
        Tuple of (feature array, expanded feature column names including _missing cols)
    """
    # Check if schema is frozen (feature_cols contains *_missing columns from training)
    has_missing_cols = any(col.endswith("_missing") for col in feature_cols)

    if has_missing_cols:
        # Inference mode: frozen schema from training
        # Separate missing indicators from base features
        missing_indicators = [c for c in feature_cols if c.endswith("_missing")]
        base_features = [c for c in feature_cols if not c.endswith("_missing")]

        # Validate that all base features exist in incoming data
        available_cols = set(df.columns)
        missing_base_features = [c for c in base_features if c not in available_cols]

        if missing_base_features:
            raise ValueError(
                f"Inference schema validation failed - {len(missing_base_features)} "
                f"required feature(s) missing: {', '.join(missing_base_features[:10])}"
                f"{'...' if len(missing_base_features) > 10 else ''}. "
                f"Expected features: {base_features}"
            )

        # Reindex to get base features (now safe - all columns exist)
        result = df.reindex(columns=base_features).copy()

        # Create missing indicators: 1 if base feature is NaN, 0 otherwise
        for missing_col in missing_indicators:
            base_col = missing_col.replace("_missing", "")
            if base_col in result.columns:
                result[missing_col] = result[base_col].isna().astype(int)
            else:
                # Base feature doesn't exist in incoming data - mark as missing
                result[missing_col] = 1
                # Also add the base column as all NaN so it can be filled with sentinel
                result[base_col] = np.nan

        # Fill NaN in base features with sentinel (but NOT the missing indicators)
        result[base_features] = result[base_features].fillna(sentinel)

        # Ensure missing indicators are int (0 or 1, never NaN or sentinel)
        for missing_col in missing_indicators:
            result[missing_col] = result[missing_col].fillna(1).astype(int)

        # Return in correct column order matching training schema
        return result[feature_cols].values, feature_cols
    else:
        # Training mode: create missing indicators dynamically
        result = df.reindex(columns=feature_cols).copy()

        # Add missing indicators for columns with NaN
        missing_cols = []
        for col in feature_cols:
            if result[col].isna().any():
                missing_col = f"{col}_missing"
                result[missing_col] = result[col].isna().astype(int)
                missing_cols.append(missing_col)

        # Fill NaN with sentinel
        result = result.fillna(sentinel)

        # Return expanded column list
        expanded_cols = feature_cols + missing_cols
        return result[expanded_cols].values, expanded_cols
