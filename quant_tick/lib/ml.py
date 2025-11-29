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


def generate_per_horizon_labels(
    df: DataFrame,
    lower_pct: float,
    upper_pct: float,
    decision_horizons: list[int],
) -> DataFrame:
    """Generate binary touch labels for range breach prediction.

    Creates simple yes/no labels for each horizon: did price touch this bound
    within the next H bars?

    Label logic per bar:
    - hit_lower_by_60: 1 if low price touches lower bound in next 60 bars, else 0
    - hit_upper_by_60: 1 if high price touches upper bound in next 60 bars, else 0
    - Repeat for each horizon (60, 120, 180, etc.)

    Example:
        If current price = 100, lower_pct = -0.03, decision_horizons = [60]:
        - Lower bound = 97
        - If price dips to 96.5 within next 60 bars: hit_lower_by_60 = 1
        - Otherwise: hit_lower_by_60 = 0

    Args:
        df: DataFrame of features
        lower_pct: Lower bound as fraction (e.g., -0.03 for -3%)
        upper_pct: Upper bound as fraction (e.g., 0.05 for +5%)
        decision_horizons: List of horizons to label (e.g., [60, 120, 180])

    Returns:
        DataFrame of features and labels
        - hit_lower_by_{h}: Binary label (0 or 1)
        - hit_upper_by_{h}: Binary label (0 or 1)
        - bar_idx, timestamp_idx: For time-series CV
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    result_rows = []
    for bar_idx in range(n - 1):
        entry_price = close[bar_idx]
        lower_bound = entry_price * (1 + lower_pct)
        upper_bound = entry_price * (1 + upper_pct)

        # Copy base row
        row = df.iloc[[bar_idx]].copy()
        row["bar_idx"] = bar_idx
        row["timestamp_idx"] = bar_idx

        # Add metadata
        row["entry_price"] = entry_price
        row["lower_bound_pct"] = lower_pct
        row["upper_bound_pct"] = upper_pct

        # Add config/bound features (critical for models to differentiate ranges)
        row["width"] = upper_pct - lower_pct
        row["asymmetry"] = upper_pct + lower_pct
        row["range_width"] = upper_pct - lower_pct
        row["range_asymmetry"] = upper_pct + lower_pct
        row["dist_to_lower_pct"] = (entry_price - lower_bound) / entry_price
        row["dist_to_upper_pct"] = (upper_bound - entry_price) / entry_price

        # For each decision horizon, check if bounds touched within next H bars
        for h in decision_horizons:
            # +1 to include exactly h bars in slice (Python slices are end-exclusive)
            end_idx = min(bar_idx + h + 1, n)

            future_lows = low[bar_idx + 1 : end_idx]
            future_highs = high[bar_idx + 1 : end_idx]
            hit_lower = (future_lows <= lower_bound).any() if len(future_lows) > 0 else False
            hit_upper = (future_highs >= upper_bound).any() if len(future_highs) > 0 else False

            row[f"hit_lower_by_{h}"] = 1 if hit_lower else 0
            row[f"hit_upper_by_{h}"] = 1 if hit_upper else 0

        result_rows.append(row)

    if len(result_rows) == 0:
        return pd.DataFrame()

    result = pd.concat(result_rows, ignore_index=True)
    return result


def generate_multi_config_labels(
    df: DataFrame,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
    decision_horizons: list[int] | None = None,
) -> DataFrame:
    """Generate per-horizon touch labels for multiple bound configurations.

    One row per bar+config, no person-period expansion. Direct binary targets per horizon.

    Args:
        df: DataFrame with features and 'close' column
        widths: List of range widths (e.g., [0.03, 0.05, 0.07])
        asymmetries: List of asymmetries (e.g., [-0.2, 0, 0.2])
        decision_horizons: List of horizons to label (e.g., [60, 120, 180])

    Returns:
        DataFrame with one row per bar+config containing per-horizon touch labels
    """
    if widths is None:
        widths = DEFAULT_WIDTHS
    if asymmetries is None:
        asymmetries = DEFAULT_ASYMMETRIES
    if decision_horizons is None:
        decision_horizons = [60, 120, 180]

    configs = [(w, a) for w in widths for a in asymmetries]

    # Generate per-horizon labels for all configs
    # Build dict of config -> labeled df first
    config_dfs = {}
    for width, asym in configs:
        lower_pct = -width * (0.5 - asym)
        upper_pct = width * (0.5 + asym)

        # Get labels for this config
        labeled = generate_per_horizon_labels(df, lower_pct, upper_pct, decision_horizons)
        config_dfs[(width, asym)] = labeled

    if len(config_dfs) == 0:
        return pd.DataFrame()

    # Interleave bar-by-bar instead of appending entire config dataframes
    # This produces ordering: [bar0_cfg0, bar0_cfg1, ..., bar1_cfg0, bar1_cfg1, ...]
    # which matches _run_backtest_loop's expectation: row_start = bar_idx * n_configs
    # Note: generate_per_horizon_labels returns n-1 rows (drops last bar with no future)
    first_config_df = next(iter(config_dfs.values()))
    n_bars = len(first_config_df)
    all_rows = []

    # Assign explicit config_id for each (width, asymmetry) pair
    config_id_map = {cfg: idx for idx, cfg in enumerate(configs)}

    for bar_idx in range(n_bars):
        for width, asym in configs:
            row = config_dfs[(width, asym)].iloc[bar_idx].copy()
            row["width"] = width
            row["asymmetry"] = asym
            row["config_id"] = config_id_map[(width, asym)]  # Explicit composite key
            all_rows.append(row)

    result = pd.DataFrame(all_rows)
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
            violations.append({
                "from_horizon": h_prev,
                "to_horizon": h_curr,
                "decrease": decrease,
                "prev_prob": horizon_probs[h_prev],
                "curr_prob": horizon_probs[h_curr],
            })

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
        logger.warning(f"Predictions lack variance ({len(unique_preds)} unique), skipping calibration")
        return None, "none"

    # Check for class imbalance (need at least a few positive samples)
    n_positive = y_true.sum()
    if n_positive < 3 or n_positive > (n_samples - 3):
        logger.warning(f"Extreme class imbalance ({n_positive}/{n_samples}), skipping calibration")
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

        logger.debug(f"Calibration ({method}): Brier {brier_before:.4f} -> {brier_after:.4f}")
        return calibrator, method

    except Exception as e:
        logger.warning(f"Calibration failed: {e}, skipping")
        return None, "none"


def apply_calibration(
    proba: float,
    calibrator: IsotonicRegression | LogisticRegression | None,
    calibration_method: str,
) -> float:
    """Apply calibrator to raw probability.

    Args:
        proba: Raw predicted probability
        calibrator: Trained calibrator (isotonic or platt)
        calibration_method: "isotonic", "platt", or "none"

    Returns:
        Calibrated probability (or raw if no calibrator)
    """
    if calibrator is None or calibration_method == "none":
        return proba

    try:
        if calibration_method == "isotonic":
            return float(calibrator.transform([proba])[0])
        elif calibration_method == "platt":
            return float(calibrator.predict_proba([[proba]])[0, 1])
        else:
            logger.warning(f"Unknown calibration method: {calibration_method}")
            return proba
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
