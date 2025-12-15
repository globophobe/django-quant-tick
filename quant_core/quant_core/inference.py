"""High-level inference API for ML predictions."""

import pandas as pd

from .constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from .prediction import (
    LPConfig,
    compute_bound_features,
    find_optimal_config,
    hazard_to_per_horizon_probs,
    prepare_features,
)


def run_inference(
    features: pd.DataFrame,
    models: dict,
    metadata: dict,
    touch_tolerance: float,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
) -> dict:
    """Run inference on features using trained models.

    This is the main entry point for prediction services (quant_horizon).
    It encapsulates the full inference workflow: feature prep, hazard prediction,
    survival curve reconstruction, and optimal config selection.

    Args:
        features: Single-row DataFrame with latest features (no k column)
        models: {"lower": lower_model, "upper": upper_model}
        metadata: {
            "feature_cols": [...],
            "max_horizon": 180,
            "decision_horizons": [60, 120, 180],
            ...
        }
        touch_tolerance: Max acceptable P(touch) for valid config
        widths: Width grid (defaults to DEFAULT_WIDTHS)
        asymmetries: Asymmetry grid (defaults to DEFAULT_ASYMMETRIES)

    Returns:
        On success:
        {
            "lower_bound": -0.03,
            "upper_bound": 0.05,
            "borrow_ratio": 0.5,
            "p_touch_lower": 0.12,
            "p_touch_upper": 0.14,
            "width": 0.04,
            "asymmetry": 0.0
        }

        On failure (no valid config):
        {
            "error": "no_valid_config",
            "message": "All configs exceed touch_tolerance=0.15"
        }
    """
    # Extract metadata
    feature_cols = metadata["feature_cols"]
    max_horizon = metadata["max_horizon"]
    decision_horizons = metadata.get("decision_horizons", [60, 120, 180])

    # Use defaults if not provided
    if widths is None:
        widths = metadata.get("widths", DEFAULT_WIDTHS)
    if asymmetries is None:
        asymmetries = metadata.get("asymmetries", DEFAULT_ASYMMETRIES)

    # Get models
    lower_model = models["lower"]
    upper_model = models["upper"]

    # Extract latest bar (should be single row)
    latest = features.iloc[[-1]].copy()

    # Create prediction functions for lower/upper bounds
    def predict_lower(feat: pd.DataFrame, lower_pct: float, upper_pct: float) -> float:
        """Predict max P(hit_lower) across decision horizons using survival model."""
        # Add bound features
        feat_with_bounds = compute_bound_features(feat, lower_pct, upper_pct)

        # Exclude k from prepare_features (not in live candle data)
        base_feature_cols = [c for c in feature_cols if c != "k"]
        X_array, expanded_cols = prepare_features(feat_with_bounds, base_feature_cols)

        # Rebuild DataFrame (no k yet)
        X_df = pd.DataFrame(X_array, columns=expanded_cols)
        X_base = X_df.iloc[[0]]

        # Reconstruct per-horizon probabilities from hazard model
        # Pass full feature_cols (includes k) - hazard_to_per_horizon_probs will add it
        horizon_probs = hazard_to_per_horizon_probs(
            lower_model,
            X_base,
            feature_cols,  # Full training order WITH k
            decision_horizons,
            max_horizon,
        )

        # Return max risk (conservative)
        return float(max(horizon_probs.values()))

    def predict_upper(feat: pd.DataFrame, lower_pct: float, upper_pct: float) -> float:
        """Predict max P(hit_upper) across decision horizons using survival model."""
        feat_with_bounds = compute_bound_features(feat, lower_pct, upper_pct)
        base_feature_cols = [c for c in feature_cols if c != "k"]
        X_array, expanded_cols = prepare_features(feat_with_bounds, base_feature_cols)
        X_df = pd.DataFrame(X_array, columns=expanded_cols)
        X_base = X_df.iloc[[0]]

        horizon_probs = hazard_to_per_horizon_probs(
            upper_model,
            X_base,
            feature_cols,  # Full training order WITH k
            decision_horizons,
            max_horizon,
        )

        return float(max(horizon_probs.values()))

    # Find optimal config
    optimal: LPConfig | None = find_optimal_config(
        predict_lower,
        predict_upper,
        latest,
        touch_tolerance=touch_tolerance,
        widths=widths,
        asymmetries=asymmetries,
    )

    if optimal is None:
        return {
            "error": "no_valid_config",
            "message": f"All configs exceed touch_tolerance={touch_tolerance}",
        }

    # Return optimal config as dict
    return {
        "lower_bound": optimal.lower_pct,
        "upper_bound": optimal.upper_pct,
        "borrow_ratio": optimal.borrow_ratio,
        "p_touch_lower": optimal.p_touch_lower,
        "p_touch_upper": optimal.p_touch_upper,
        "width": optimal.width,
        "asymmetry": optimal.asymmetry,
    }
