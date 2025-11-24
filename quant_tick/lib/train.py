import hashlib
import logging
from io import BytesIO
from typing import Any

import joblib
import numpy as np
from django.core.files.base import ContentFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from quant_tick.lib.ml import PurgedKFold
from quant_tick.models import MLArtifact, MLConfig, MLFeatureData

logger = logging.getLogger(__name__)


def compute_feature_hash(feature_cols: list[str]) -> str:
    """Compute hash of feature columns for drift detection."""
    cols_str = ",".join(sorted(feature_cols))
    return hashlib.sha256(cols_str.encode()).hexdigest()[:16]


# Feature columns to exclude from training
EXCLUDE_COLS = [
    "timestamp",
    "timestamp_idx",
    "touched_lower",
    "touched_upper",
    "entry_price",
    "lower_bound_pct",
    "upper_bound_pct",
    "horizon_bars",
]


def train_models(
    config: MLConfig,
    n_splits: int = 5,
    embargo_bars: int = 96,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 50,
    max_features: str = "sqrt",
) -> bool:
    """Train ML models for touch prediction.

    Args:
        config: MLConfig to train models for
        n_splits: Number of CV folds
        embargo_bars: Embargo period in bars
        n_estimators: Number of trees
        max_depth: Max tree depth
        min_samples_leaf: Minimum samples per leaf
        max_features: Max features per split
    """
    # Get latest feature data
    feature_data = MLFeatureData.objects.filter(
        candle=config.candle
    ).order_by("-timestamp_to").first()

    if not feature_data or not feature_data.has_data_frame("file_data"):
        logger.error(f"{config}: no feature data available")
        return False

    df = feature_data.get_data_frame("file_data")
    logger.info(f"{config}: loaded {len(df)} rows from feature data")

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    logger.info(f"{config}: using {len(feature_cols)} features")

    # Prepare data - RF doesn't handle NaN, fill with 0
    X = df[feature_cols].fillna(0).values
    y_lower = df["touched_lower"].values
    y_upper = df["touched_upper"].values

    # Get horizon from config
    horizon_bars = config.horizon_bars

    # Event end indices for purging using timestamp_idx
    timestamp_idx = df["timestamp_idx"].values
    event_end_idx = timestamp_idx + horizon_bars

    logger.info(
        f"{config}: lower_rate={y_lower.mean():.3f}, upper_rate={y_upper.mean():.3f}"
    )

    # Train with PurgedKFold cross-validation
    cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)

    # RF params
    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    brier_lower_scores = []
    brier_upper_scores = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, event_end_idx=event_end_idx, timestamp_idx=timestamp_idx)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_lower_train, y_lower_test = y_lower[train_idx], y_lower[test_idx]
        y_upper_train, y_upper_test = y_upper[train_idx], y_upper[test_idx]

        logger.info(
            f"{config}: fold {fold+1}/{n_splits} - "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

        # Train models
        model_lower = RandomForestClassifier(**rf_params)
        model_lower.fit(X_train, y_lower_train)

        model_upper = RandomForestClassifier(**rf_params)
        model_upper.fit(X_train, y_upper_train)

        # Evaluate
        p_lower_test = model_lower.predict_proba(X_test)[:, 1]
        p_upper_test = model_upper.predict_proba(X_test)[:, 1]

        brier_lower = brier_score_loss(y_lower_test, p_lower_test)
        brier_upper = brier_score_loss(y_upper_test, p_upper_test)

        brier_lower_scores.append(brier_lower)
        brier_upper_scores.append(brier_upper)

        logger.info(
            f"{config}: fold {fold+1} Brier - "
            f"lower={brier_lower:.4f}, upper={brier_upper:.4f}"
        )

    avg_brier_lower = np.mean(brier_lower_scores)
    avg_brier_upper = np.mean(brier_upper_scores)

    logger.info(
        f"{config}: CV avg Brier - "
        f"lower={avg_brier_lower:.4f}, upper={avg_brier_upper:.4f}"
    )

    # Train final models on all data
    logger.info(f"{config}: training final models on all data")

    final_model_lower = RandomForestClassifier(**rf_params)
    final_model_lower.fit(X, y_lower)

    final_model_upper = RandomForestClassifier(**rf_params)
    final_model_upper.fit(X, y_upper)

    # Compute feature hash for drift detection
    feature_hash = compute_feature_hash(feature_cols)

    # Save models
    _save_artifact(
        config, final_model_lower, "lower", feature_cols, avg_brier_lower, feature_hash
    )
    _save_artifact(
        config, final_model_upper, "upper", feature_cols, avg_brier_upper, feature_hash
    )

    logger.info(f"{config}: training complete (feature_hash={feature_hash})")
    return True


def _save_artifact(
    config: MLConfig,
    model: Any,
    model_type: str,
    feature_cols: list[str],
    brier_score: float,
    feature_hash: str,
) -> None:
    """Save model artifact with feature hash for drift detection."""
    # Delete existing artifacts of same type
    MLArtifact.objects.filter(
        ml_config=config,
        model_type=model_type,
    ).delete()

    # Serialize model
    buf = BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)

    artifact = MLArtifact(
        ml_config=config,
        model_type=model_type,
        brier_score=brier_score,
        feature_columns=feature_cols,
    )
    artifact.artifact.save(
        f"ml_{model_type}.joblib",
        ContentFile(buf.read()),
    )
    # Store feature hash in json_data on config for drift detection
    if "feature_hashes" not in config.json_data:
        config.json_data["feature_hashes"] = {}
    config.json_data["feature_hashes"][model_type] = feature_hash
    config.save(update_fields=["json_data"])

    artifact.save()

    logger.info(f"{config}: saved {model_type} model artifact (feature_hash={feature_hash})")
