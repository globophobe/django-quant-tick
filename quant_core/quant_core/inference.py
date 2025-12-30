import pandas as pd

from .constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from .prediction import predict_competing_risks_multi_horizon


def run_inference(
    features: pd.DataFrame,
    models: dict,
    metadata: dict,
    touch_tolerance: float,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
) -> dict:
    """Run inference using competing-risks models.

    This is the main entry point for prediction services (quant_horizon).

    Args:
        features: Single-row DataFrame with latest features
        models: {"first_hit_h48": model, "first_hit_h96": model, ...}
        metadata: {
            "feature_cols": [...],
            "horizons": [48, 96, 144],
            "model_kind": "competing_risks",
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
    # Validate model_kind
    model_kind = metadata.get("model_kind")
    if model_kind != "competing_risks":
        raise ValueError(
            f"Unsupported model_kind='{model_kind}'. "
            "Only 'competing_risks' models are supported."
        )

    feature_cols = metadata["feature_cols"]
    horizons = metadata["horizons"]

    if widths is None:
        widths = metadata.get("widths", DEFAULT_WIDTHS)
    if asymmetries is None:
        asymmetries = metadata.get("asymmetries", DEFAULT_ASYMMETRIES)

    # Use shortest horizon for risk assessment and scoring
    decision_horizon = min(horizons)

    latest = features.iloc[[-1]].copy()
    configs = [(w, a) for w in widths for a in asymmetries]
    valid_configs = []

    for width, asymmetry in configs:
        lower_pct = -width * (0.5 - asymmetry)
        upper_pct = width * (0.5 + asymmetry)

        # Get predictions for all horizons
        preds = predict_competing_risks_multi_horizon(
            models,
            latest,
            horizons,
            lower_pct,
            upper_pct,
            width,
            asymmetry,
            feature_cols,
        )

        p_touch_lower = preds[decision_horizon]["DOWN_FIRST"]
        p_touch_upper = preds[decision_horizon]["UP_FIRST"]
        p_timeout = preds[decision_horizon]["TIMEOUT"]

        # In competing-risks: P(exit) = P(DOWN_FIRST) + P(UP_FIRST) = 1 - P(TIMEOUT)
        p_exit = p_touch_lower + p_touch_upper
        if p_exit <= touch_tolerance:
            valid_configs.append(
                {
                    "lower_pct": lower_pct,
                    "upper_pct": upper_pct,
                    "p_touch_lower": p_touch_lower,
                    "p_touch_upper": p_touch_upper,
                    "width": width,
                    "asymmetry": asymmetry,
                    "p_timeout": p_timeout,
                }
            )

    if not valid_configs:
        return {
            "error": "no_valid_config",
            "message": f"All configs exceed touch_tolerance={touch_tolerance}",
        }

    # Select config with highest P(TIMEOUT) at decision_horizon
    best = max(valid_configs, key=lambda x: x["p_timeout"])

    return {
        "lower_bound": best["lower_pct"],
        "upper_bound": best["upper_pct"],
        "borrow_ratio": 0.5 + best["asymmetry"],
        "p_touch_lower": best["p_touch_lower"],
        "p_touch_upper": best["p_touch_upper"],
        "width": best["width"],
        "asymmetry": best["asymmetry"],
    }
