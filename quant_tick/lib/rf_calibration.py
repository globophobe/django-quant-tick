"""Random Forest Probability Calibration - Making Probabilities Actually Mean Something

Random forests are great at ranking predictions (this sample is riskier than that one),
but their raw probabilities are often poorly calibrated. A prediction of 70% might
actually happen 85% of the time, or only 55% of the time.

Calibration fixes this: it adjusts the probabilities so that when the model says 70%,
it really means 70%. This is critical for any application where you use the probability
value itself (not just the ranking) - like position sizing, betting, or cost-sensitive
decisions.

Calibration methods:
- Platt scaling (sigmoid): Fits a logistic regression on top of the model's outputs.
  Works well for small datasets and fixes systematic over/under-confidence.
- Isotonic regression: Non-parametric method that can fix any monotonic miscalibration.
  Better for large datasets but can overfit on small ones.
- Calibration curves: Visual check showing predicted probability vs actual frequency.
  Perfect calibration = diagonal line.
- Brier score: Measures squared error of probabilities (lower = better calibrated).
- ECE: Expected Calibration Error, average absolute difference between predicted and
  actual probabilities across bins.

Technical implementation follows mlbook.explained.ai using sklearn.calibration with
the cv="prefit" pattern for post-training calibration.
"""

import logging
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def calibrate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "sigmoid",
    sample_weight: np.ndarray | None = None,
) -> CalibratedClassifierCV:
    """Adjust model probabilities to match actual frequencies.

    Takes a trained model and learns a simple transformation (sigmoid or isotonic) that
    maps the model's raw probabilities to calibrated probabilities. For example, if the
    model outputs 0.6 but the actual positive rate for those predictions is 0.75, the
    calibration layer learns to map 0.6 → 0.75.

    Uses cv="prefit" pattern: the model is already trained, we just fit the calibration
    layer on top using the same data (or ideally out-of-fold predictions).

    Multi-class support: For multi-class problems (e.g., {-1, 0, +1} labels), sklearn
    applies calibration in one-vs-rest fashion automatically. Each class gets its own
    calibration. To generate calibration curves/metrics for a specific class, the caller
    should convert labels to binary (target_class vs rest) before calling curve/metric
    functions.

    Args:
        model: Already trained RandomForestClassifier
        X: Feature matrix used for calibration (samples × features)
        y: True labels (binary or multi-class)
        method: "sigmoid" for Platt scaling or "isotonic" for isotonic regression
        sample_weight: Optional weights for each sample

    Returns:
        CalibratedClassifierCV wrapper - use this for predictions instead of raw model
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X, y, sample_weight=sample_weight)
    return calibrated


def generate_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Generate calibration curve data (JSON-serializable).

    For multi-class problems, caller should convert y_true to binary (target_class vs rest)
    and pass the probability for that specific class.

    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for the positive class (single column)
        n_bins: Number of bins for calibration curve

    Returns:
        Dict with bin_edges, bin_frequencies, bin_counts, perfect_calibration
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts = []
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        counts.append(int(np.sum(mask)))

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_midpoints": bin_midpoints.tolist(),
        "bin_frequencies": prob_true.tolist(),
        "bin_predictions": prob_pred.tolist(),
        "bin_counts": counts,
        "perfect_calibration": bin_midpoints.tolist(),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Measure how well calibrated the probabilities are.

    Brier score: Mean squared error between predicted probabilities and actual outcomes.
    Lower is better, perfect score is 0. Penalizes both over-confidence and under-confidence.

    ECE (Expected Calibration Error): Bins predictions by probability, then measures
    average gap between predicted probability and actual frequency in each bin.
    Lower is better, perfect calibration = 0.

    For multi-class problems, caller should convert y_true to binary (target_class vs rest)
    and pass the probability for that specific class.

    Args:
        y_true: Actual binary outcomes (0 or 1)
        y_proba: Predicted probabilities for the positive class (single column)

    Returns:
        Dict with keys:
        - brier_score: Squared error metric (0 = perfect)
        - ece: Average calibration error across bins (0 = perfect)
    """
    brier = float(brier_score_loss(y_true, y_proba))

    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="uniform"
    )
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    return {
        "brier_score": brier,
        "ece": ece,
    }


def plot_calibration_curve(
    calibration_data: dict,
    title: str = "Calibration Curve",
) -> BytesIO:
    """Plot calibration curve from curve data.

    Args:
        calibration_data: Dict from generate_calibration_curve()
        title: Plot title

    Returns:
        BytesIO buffer with PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    bin_predictions = calibration_data["bin_predictions"]
    bin_frequencies = calibration_data["bin_frequencies"]

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

    ax.plot(
        bin_predictions,
        bin_frequencies,
        "s-",
        label="Model calibration",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf
