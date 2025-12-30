import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .constants import MISSING_SENTINEL

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
    direction: str | None = None
    exposure: float = 0.0


def apply_multiclass_calibration(
    y_pred_proba: np.ndarray,
    calibrators: dict | None,
    methods: dict | None,
) -> np.ndarray:
    """Apply One-vs-Rest calibration to multiclass predictions.

    CRITICAL: This lives in quant_core (not quant_tick) to respect package boundaries.
    quant_core is Django/FastAPI-free, so cannot import quant_tick.

    Args:
        y_pred_proba: Raw probabilities (n_samples, n_classes)
        calibrators: Dict mapping class_idx -> calibrator (can be None per-class)
        methods: Dict mapping class_idx -> method string ("isotonic", "platt", "none")

    Returns:
        Calibrated probabilities (n_samples, n_classes), rows sum to 1.0
    """
    if calibrators is None or methods is None:
        return y_pred_proba

    n_samples, n_classes = y_pred_proba.shape
    calibrated = np.zeros_like(y_pred_proba)

    # Apply per-class calibration (handles missing keys gracefully)
    for cls in range(n_classes):
        calibrator = calibrators.get(cls)  # None if missing
        method = methods.get(cls, "none")  # "none" if missing

        if calibrator is not None and method != "none":
            # Apply calibration using binary apply_calibration
            calibrated[:, cls] = apply_calibration(
                y_pred_proba[:, cls], calibrator, method
            )
        else:
            # Skip calibration for this class (graceful degradation)
            calibrated[:, cls] = y_pred_proba[:, cls]

    # Re-normalize to sum to 1.0 (required for valid probabilities)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return calibrated / row_sums


def predict_competing_risks_multi_horizon(
    models_dict: dict[str, Any],
    features: pd.DataFrame,
    horizons: list[int],
    lower_pct: float,
    upper_pct: float,
    width: float,
    asymmetry: float,
    feature_cols: list[str],
) -> dict[int, dict[str, float]]:
    """Predict first-hit probabilities for competing-risks models.

    Args:
        models_dict: {"first_hit_h48": model, "first_hit_h96": model, ...}
        features: Single-row DataFrame with base features
        horizons: List of horizons (e.g., [48, 96, 144])
        lower_pct: Lower barrier offset (e.g., -0.03)
        upper_pct: Upper barrier offset (e.g., +0.03)
        width: Config width (must match training)
        asymmetry: Config asymmetry (must match training)
        feature_cols: Feature column names

    Returns:
        {
            48: {"UP_FIRST": 0.25, "DOWN_FIRST": 0.20, "TIMEOUT": 0.55},
            96: {"UP_FIRST": 0.30, "DOWN_FIRST": 0.25, "TIMEOUT": 0.45},
            ...
        }
    """
    # Add config features (must match training exactly)
    feat_with_bounds = compute_bound_features(features, lower_pct, upper_pct)
    feat_with_bounds["width"] = width
    feat_with_bounds["asymmetry"] = asymmetry

    X_array, expanded_cols = prepare_features(feat_with_bounds, feature_cols)
    X_df = pd.DataFrame(X_array, columns=expanded_cols)

    results = {}
    for H in horizons:
        model_key = f"first_hit_h{H}"
        if model_key not in models_dict:
            raise ValueError(f"Model {model_key} not found in models_dict")

        model = models_dict[model_key]

        # predict_proba returns [P(UP_FIRST), P(DOWN_FIRST), P(TIMEOUT)]
        probs = model.predict_proba(X_df)[0]

        # CRITICAL: Always remap predict_proba to fixed 3-class layout using model.classes_
        # LightGBM's predict_proba returns columns in model.classes_ order, NOT [0,1,2] order
        # Even with 3 classes, classes_ could be [0,2,1] if training data happened to see class 0 first
        # Must always map to canonical [0:UP_FIRST, 1:DOWN_FIRST, 2:TIMEOUT] layout
        if hasattr(model, "classes_"):
            probs_full = np.zeros(3)
            for i, cls in enumerate(model.classes_):
                probs_full[cls] = probs[i]
            probs = probs_full

        # Apply calibration if available (optional, graceful degradation)
        if hasattr(model, "calibrator_") and model.calibrator_ is not None:
            calibration_methods = getattr(model, "calibration_methods_", None)

            # Only apply if both calibrator and methods exist
            if calibration_methods is not None:
                probs_2d = probs.reshape(1, -1)
                calibrated_2d = apply_multiclass_calibration(
                    probs_2d, model.calibrator_, calibration_methods
                )
                probs = calibrated_2d[0]

        results[H] = {
            "UP_FIRST": float(probs[0]),
            "DOWN_FIRST": float(probs[1]),
            "TIMEOUT": float(probs[2]),
        }

    return results


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

        # Reindex to ensure all base features exist (fills missing ones with NaN)
        result = df.reindex(columns=base_features).copy()

        # Compute missing mask for all base features at once
        missing_mask = result.isna()

        # Build all missing indicator columns as a DataFrame
        import pandas as pd

        missing_df = missing_mask.astype(np.int8)
        missing_df.columns = [f"{col}_missing" for col in missing_df.columns]

        # Reindex to include all expected missing indicators
        missing_df = missing_df.reindex(
            columns=missing_indicators, fill_value=1
        ).astype(np.int8)

        # Fill NaN with sentinel in base features
        result = result.fillna(sentinel)

        # Concatenate missing indicators in one operation
        result = pd.concat([result, missing_df], axis=1)

        # Return in correct column order matching training schema
        return result[feature_cols].values, feature_cols
    else:
        # Training mode: create missing indicators dynamically
        result = df.reindex(columns=feature_cols).copy()

        # Identify columns with NaN values
        cols_with_nan = [col for col in feature_cols if result[col].isna().any()]

        if cols_with_nan:
            # Build missing indicators for columns with NaN
            missing_mask = result[cols_with_nan].isna()
            missing_df = missing_mask.astype(np.int8)
            missing_df.columns = [f"{col}_missing" for col in missing_df.columns]

            # Fill NaN with sentinel
            result = result.fillna(sentinel)

            # Concatenate missing indicators in one operation
            result = pd.concat([result, missing_df], axis=1)

            missing_cols = list(missing_df.columns)
            expanded_cols = feature_cols + missing_cols
            return result[expanded_cols].values, expanded_cols
        else:
            # No missing values - just fill with sentinel
            result = result.fillna(sentinel)
            return result[feature_cols].values, feature_cols
