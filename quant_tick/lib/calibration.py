import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def apply_calibration(
    y_pred_proba: np.ndarray, calibrator: object, method: str
) -> np.ndarray:
    """Apply a calibration model to predicted probabilities."""
    if method == "platt":
        return calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.transform(y_pred_proba)
    return y_pred_proba


def calibrate_probabilities(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    method: str = "auto",
    min_samples: int = 10,
    min_positive: int = 3,
) -> tuple[IsotonicRegression | LogisticRegression | None, str]:
    """Fit a calibrator and validate that calibration improves Brier score."""
    n_samples = len(y_true)
    if n_samples < min_samples:
        logger.warning("Too few samples for calibration (%s), skipping", n_samples)
        return None, "none"

    unique_preds = np.unique(y_pred_proba)
    if len(unique_preds) < 3:
        logger.warning(
            "Predictions lack variance (%s unique), skipping calibration",
            len(unique_preds),
        )
        return None, "none"

    positives = int(np.sum(y_true))
    if positives < min_positive or positives > (n_samples - min_positive):
        logger.warning(
            "Class imbalance too extreme (%s/%s), skipping calibration",
            positives,
            n_samples,
        )
        return None, "none"

    if method == "auto":
        method = "isotonic" if n_samples >= 50 else "platt"

    try:
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_pred_proba, y_true)
            y_calibrated = calibrator.transform(y_pred_proba)
        elif method == "platt":
            calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            calibrator.fit(y_pred_proba.reshape(-1, 1), y_true)
            y_calibrated = calibrator.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
        else:
            return None, "none"

        brier_before = brier_score_loss(y_true, y_pred_proba)
        brier_after = brier_score_loss(y_true, y_calibrated)
        if brier_after > brier_before * 1.05:
            logger.warning(
                "Calibration worsened Brier score (%.6f -> %.6f), skipping",
                brier_before,
                brier_after,
            )
            return None, "none"
    except Exception as exc:
        logger.warning("Calibration failed: %s", exc)
        return None, "none"

    return calibrator, method
