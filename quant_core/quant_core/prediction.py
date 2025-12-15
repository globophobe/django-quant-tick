"""Core prediction logic for hazard-based survival analysis."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS, MISSING_SENTINEL

logger = logging.getLogger(__name__)


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
