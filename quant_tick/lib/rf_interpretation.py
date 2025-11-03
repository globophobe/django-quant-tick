"""Random Forest Interpretation - Understanding What the Model Actually Learned

This module helps you see inside the black box. Instead of just getting predictions,
you can understand WHY the model made those predictions and HOW each feature influences
the output.

Core interpretation methods:
- Partial Dependence Plots (PDP): Shows how changing one feature affects predictions
  while averaging over all other features. Answers "what happens if I increase X?"
- ICE Plots: Like PDP but shows individual predictions instead of averages, revealing
  interactions and non-linear effects that PDP might hide.
- TreeInterpreter: Breaks down each individual prediction into contributions from
  each feature. Shows exactly which features pushed the prediction up or down.
- SHAP: Similar to TreeInterpreter but with better theoretical properties. Shows
  feature importance and direction of impact in a single visualization.

Technical implementation follows mlbook.explained.ai and fastai recommendations using
sklearn.inspection, TreeInterpreter, and SHAP libraries.
"""
import logging
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from treeinterpreter import treeinterpreter as ti

logger = logging.getLogger(__name__)


def generate_pdp_plots(
    model: RandomForestClassifier,
    X: np.ndarray,
    feature_names: list[str],
    top_k_features: int = 10,
    kind: str = "both",
) -> dict[str, BytesIO]:
    """Create plots showing how predictions change as you vary each feature.

    Partial Dependence Plots answer: "If I change feature X while keeping everything
    else average, how does the prediction change?" This reveals the feature's overall
    trend and whether the relationship is linear, curved, or has thresholds.

    ICE (Individual Conditional Expectation) plots show the same thing but for each
    individual sample instead of averaging. This reveals heterogeneity - maybe the
    feature matters a lot for some samples but not others.

    Args:
        model: Trained RandomForestClassifier
        X: Feature matrix used to compute dependence (samples × features)
        feature_names: Names of each feature column
        top_k_features: How many features to plot (plots the most important ones)
        kind: "average" for PDP only, "individual" for ICE only, "both" for both

    Returns:
        Dictionary mapping feature name → PNG plot buffer
    """
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_k_features]

    plot_buffers = {}

    for idx in top_indices:
        if idx >= len(feature_names):
            continue

        feature_name = feature_names[idx]

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            PartialDependenceDisplay.from_estimator(
                model,
                X,
                features=[idx],
                kind=kind,
                ax=ax,
                ice_lines_kw={"alpha": 0.3, "linewidth": 0.5} if kind in ["both", "individual"] else None,
                pd_line_kw={"linewidth": 2, "color": "red"} if kind in ["both", "average"] else None,
            )

            ax.set_title(f"Partial Dependence: {feature_name}")
            ax.set_xlabel(feature_name)

            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            plot_buffers[feature_name] = buf

        except Exception as e:
            logger.warning(f"Failed to generate PDP for {feature_name}: {e}")
            continue

    return plot_buffers


def compute_tree_contributions(
    model: RandomForestClassifier,
    X: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, float]]:
    """Break down each prediction into contributions from individual features.

    For each prediction, this shows you exactly how much each feature pushed the
    prediction up or down. If the prediction is 0.75 and feature_A contributed +0.2
    while feature_B contributed -0.1, you know feature_A was strongly positive and
    feature_B was weakly negative for that specific sample.

    This is extremely useful for debugging predictions and understanding why the model
    made a particular decision.

    Args:
        model: Trained RandomForestClassifier
        X: Feature matrix for samples you want to explain (samples × features)
        feature_names: Names of each feature column

    Returns:
        List of dicts (one per sample) mapping feature name → contribution value
        Positive = pushed prediction up, negative = pushed it down
    """
    prediction, bias, contributions = ti.predict(model, X)

    results = []
    for sample_contributions in contributions:
        contrib_dict = {
            feature_names[i]: float(sample_contributions[i, 1])
            if sample_contributions.shape[1] > 1
            else float(sample_contributions[i])
            for i in range(len(feature_names))
            if i < len(sample_contributions)
        }
        results.append(contrib_dict)

    return results


def generate_shap_summary(
    model: RandomForestClassifier,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 100,
) -> BytesIO:
    """Create a summary visualization of feature impacts using SHAP values.

    SHAP (SHapley Additive exPlanations) shows two things at once:
    1. Which features are most important overall (vertical position)
    2. Whether high feature values increase or decrease predictions (color)

    The plot is a dot cloud where each dot is one sample. Red dots are high feature
    values, blue dots are low feature values. If red dots are consistently on the right,
    that feature increases predictions when it's high.

    SHAP is computationally expensive, so we limit to max_samples.

    Args:
        model: Trained RandomForestClassifier
        X: Feature matrix (will use up to max_samples)
        feature_names: Names of each feature column
        max_samples: Max samples to compute SHAP for (keeps it fast)

    Returns:
        PNG plot buffer with SHAP summary visualization
    """
    X_sample = X[:max_samples] if len(X) > max_samples else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def save_interpretation_artifacts(
    pdp_plots: dict[str, BytesIO],
    shap_summary: BytesIO | None,
    ml_run: Any,
    artifact_type: str = "primary_model",
) -> list:
    """Save interpretation plots as MLArtifact files.

    Args:
        pdp_plots: Dictionary of feature name to PDP plot buffer
        shap_summary: Optional SHAP summary plot buffer
        ml_run: MLRun instance
        artifact_type: Type of artifact ("primary_model" or "meta_model")

    Returns:
        List of created MLArtifact instances
    """
    from django.core.files.base import ContentFile

    from quant_tick.models import MLArtifact

    artifacts = []

    for feature_name, buf in pdp_plots.items():
        safe_name = feature_name.replace("/", "_").replace(" ", "_")
        filename = f"pdp_{safe_name}.png"

        artifact = MLArtifact.objects.create(
            ml_run=ml_run,
            artifact=ContentFile(buf.read(), filename),
            artifact_type=f"{artifact_type}_pdp",
            version="1.0",
        )
        artifacts.append(artifact)

    if shap_summary:
        artifact = MLArtifact.objects.create(
            ml_run=ml_run,
            artifact=ContentFile(shap_summary.read(), "shap_summary.png"),
            artifact_type=f"{artifact_type}_shap",
            version="1.0",
        )
        artifacts.append(artifact)

    return artifacts


def get_feature_contribution_summary(
    contributions: list[dict[str, float]],
    top_k: int = 10,
) -> dict:
    """Summarize feature contributions across multiple predictions.

    Args:
        contributions: List of per-prediction contribution dicts
        top_k: Number of top features to include

    Returns:
        Summary dict with mean absolute contributions and top features
    """
    if not contributions:
        return {"top_features": [], "mean_abs_contributions": {}}

    all_features = set()
    for contrib in contributions:
        all_features.update(contrib.keys())

    mean_abs_contributions = {}
    for feature in all_features:
        values = [abs(contrib.get(feature, 0.0)) for contrib in contributions]
        mean_abs_contributions[feature] = float(np.mean(values))

    sorted_features = sorted(
        mean_abs_contributions.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "top_features": [
            {"feature": name, "mean_abs_contribution": value}
            for name, value in sorted_features[:top_k]
        ],
        "mean_abs_contributions": mean_abs_contributions,
    }
