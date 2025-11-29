import hashlib
import logging
import pickle
from io import BytesIO
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from django.core.files.base import ContentFile

from quant_tick.lib.ml import (
    PurgedKFold,
    prepare_features,
)
from quant_tick.models import MLArtifact, MLConfig, MLFeatureData

logger = logging.getLogger(__name__)


def compute_feature_hash(feature_cols: list[str]) -> str:
    """Compute hash of feature columns for drift detection."""
    cols_str = ",".join(sorted(feature_cols))
    return hashlib.sha256(cols_str.encode()).hexdigest()[:16]


# Feature columns to exclude from training (per-horizon direct classifiers)
# NOTE: width, asymmetry, and bound features are NOW INCLUDED as features
# This allows models to differentiate between different range configurations
EXCLUDE_COLS = [
    "timestamp",
    "timestamp_idx",
    "bar_idx",
    "config_id",  # Explicit config identifier (metadata only)
    "entry_price",
]


def _get_exclude_cols_for_df(df: pd.DataFrame) -> list[str]:
    """Get list of columns to exclude from features, including per-horizon targets."""
    exclude = list(EXCLUDE_COLS)

    # Dynamically add per-horizon target columns
    for col in df.columns:
        if col.startswith("hit_lower_by_") or col.startswith("hit_upper_by_"):
            exclude.append(col)

    return exclude


def train_model_core(
    df: Any,
    decision_horizons: list[int],
    n_splits: int = 5,
    embargo_bars: int = 96,
    n_estimators: int = 300,
    max_depth: int = 6,
    min_samples_leaf: int = 50,
    learning_rate: float = 0.05,
    subsample: float = 0.75,
    holdout_pct: float = 0.2,
    calibration_pct: float = 0.1,
) -> tuple[dict[str, Any], list[str], dict, dict]:
    """Train per-horizon binary classifiers for range touch prediction.

    Trains separate LightGBM classifiers for each decision horizon and side:
    - lower_h60, lower_h120, lower_h180: P(lower bound touched by horizon H)
    - upper_h60, upper_h120, upper_h180: P(upper bound touched by horizon H)

    Each classifier is a simple binary model: given current features, will this
    bound get touched within the next H bars? (Yes=1, No=0)

    Process:
    1. Purged K-fold cross-validation (prevents lookahead bias)
    2. Train LightGBM with early stopping
    3. Calibrate on holdout set (isotonic or Platt)
    4. Return models + metrics

    This is NOT a hazard model or survival model. It's just per-horizon
    logistic regression with trees.

    Args:
        df: Feature dataframe with per-horizon binary targets (hit_lower_by_H, etc)
        decision_horizons: Horizons to train models for (e.g., [60, 120, 180])
        n_splits: Number of cross-validation folds
        embargo_bars: Buffer between train/test to prevent leakage
        n_estimators: Number of boosting iterations
        max_depth: Max tree depth
        min_samples_leaf: Minimum samples per leaf
        learning_rate: Boosting learning rate
        subsample: Fraction of samples for each tree
        holdout_pct: Fraction of data held out for final calibration

    Returns:
        Tuple of (models_dict, feature_cols, cv_metrics, holdout_metrics)
        - models_dict: {model_key: trained_model} with calibrator attached as .calibrator_
        - feature_cols: List of feature names used
        - cv_metrics: Cross-validation Brier scores
        - holdout_metrics: Holdout Brier scores and calibration gaps
    """
    # Identify feature columns (exclude targets and metadata)
    exclude_cols = _get_exclude_cols_for_df(df)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Prepare features with missing indicators and sentinel fill
    X, feature_cols = prepare_features(df, feature_cols)

    # Get timestamp for splitting
    timestamp_idx = df["bar_idx"].values  # Use bar_idx as timestamp

    # Three-way split at bar-level with horizon-aware purging
    # Determine number of configs per bar
    unique_bar_indices = df["bar_idx"].unique()
    n_bars = len(unique_bar_indices)
    n_configs = len(df) // n_bars

    # Get max horizon for purging (labels depend on prices up to bar_idx + max_horizon)
    max_horizon = max(decision_horizons)

    # Minimum bar requirements
    MIN_BARS_PER_SPLIT = 5

    # Initial bar-based split (before purging)
    test_bars = int(n_bars * holdout_pct)
    calib_bars = int(n_bars * calibration_pct)
    train_bars_initial = n_bars - test_bars - calib_bars

    # Purge training samples whose event horizon overlaps with calib/test
    # Training can only use bars where bar_idx + max_horizon + 1 <= first_calib_bar
    first_calib_bar = train_bars_initial
    first_test_bar = train_bars_initial + calib_bars

    # Purge training: exclude samples within max_horizon of calib
    train_bars_purged = max(0, train_bars_initial - (max_horizon + 1))

    # Purge calibration: exclude samples within max_horizon of test
    calib_bars_purged = max(0, calib_bars - (max_horizon + 1))

    # Check minimum requirements after purging
    if train_bars_purged < MIN_BARS_PER_SPLIT:
        raise ValueError(
            f"After purging with horizon={max_horizon}, train has only {train_bars_purged} bars "
            f"(need {MIN_BARS_PER_SPLIT}). Total bars: {n_bars}. "
            f"Collect more data or reduce horizons."
        )
    if calib_bars_purged < MIN_BARS_PER_SPLIT:
        raise ValueError(
            f"After purging with horizon={max_horizon}, calib has only {calib_bars_purged} bars "
            f"(need {MIN_BARS_PER_SPLIT}). Total bars: {n_bars}. "
            f"Collect more data or reduce horizons/calibration_pct."
        )
    if test_bars < MIN_BARS_PER_SPLIT:
        raise ValueError(
            f"Test has only {test_bars} bars (need {MIN_BARS_PER_SPLIT}). "
            f"Total bars: {n_bars}. Increase holdout_pct or collect more data."
        )

    # Convert bar-level split to row indices (blocks of n_configs)
    # Training uses only purged bars
    train_idx = np.arange(train_bars_purged * n_configs)

    # Calibration starts at first_calib_bar, uses only purged bars
    calib_start_row = first_calib_bar * n_configs
    calib_idx = np.arange(calib_start_row, calib_start_row + calib_bars_purged * n_configs)

    # Test starts at first_test_bar, uses all test bars (no purging needed after test)
    test_start_row = first_test_bar * n_configs
    test_idx = np.arange(test_start_row, test_start_row + test_bars * n_configs)

    # Verify no bar_idx overlap
    train_bar_set = set(df.iloc[train_idx]["bar_idx"].unique())
    calib_bar_set = set(df.iloc[calib_idx]["bar_idx"].unique())
    test_bar_set = set(df.iloc[test_idx]["bar_idx"].unique())
    assert len(train_bar_set & calib_bar_set) == 0, "Train/calib overlap detected"
    assert len(train_bar_set & test_bar_set) == 0, "Train/test overlap detected"
    assert len(calib_bar_set & test_bar_set) == 0, "Calib/test overlap detected"

    X_train_full = X[train_idx]
    X_calib = X[calib_idx]
    X_test = X[test_idx]
    timestamp_idx_train = timestamp_idx[train_idx]

    logger.info(
        f"Purged split (horizon={max_horizon}): {train_bars_purged} train bars "
        f"(from {train_bars_initial}), {calib_bars_purged} calib bars (from {calib_bars}), "
        f"{test_bars} test bars ({n_configs} configs/bar)"
    )

    # LightGBM parameters
    lgbm_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_samples": min_samples_leaf,
        "subsample": subsample,
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Train models per horizon+side
    models_dict = {}
    calibrators_dict = {}
    calibration_methods_dict = {}
    cv_brier_scores = {}
    holdout_brier_scores = {}
    holdout_calibration_gaps = {}
    base_rates = {}  # Training base rates for rare event monitoring

    for h in decision_horizons:
        for side in ["lower", "upper"]:
            label_col = f"hit_{side}_by_{h}"
            model_key = f"{side}_h{h}"

            logger.info(f"Training {model_key}...")

            # Get target variables
            y_train_full = df.iloc[train_idx][label_col].values
            y_calib = df.iloc[calib_idx][label_col].values
            y_test = df.iloc[test_idx][label_col].values

            # Compute training base rate for rare event monitoring
            train_base_rate = float(np.mean(y_train_full))
            base_rates[model_key] = train_base_rate

            # Cross-validation
            cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
            cv_brier_fold_scores = []

            # Create event end timestamps for purging
            # +1 to match corrected label horizon (labels depend on prices up to bar_idx + h)
            event_end_idx = timestamp_idx_train + h + 1

            fold_list = list(cv.split(
                X_train_full,
                event_end_idx=event_end_idx,
                timestamp_idx=timestamp_idx_train,
            ))

            if len(fold_list) == 0:
                raise ValueError(
                    f"PurgedKFold yielded zero folds for {model_key} with n_splits={n_splits}, "
                    f"embargo_bars={embargo_bars}. Purging/embargo too aggressive for dataset size."
                )

            for train_fold_idx, test_fold_idx in fold_list:
                X_cv_train = X_train_full[train_fold_idx]
                y_cv_train = y_train_full[train_fold_idx]
                X_cv_test = X_train_full[test_fold_idx]
                y_cv_test = y_train_full[test_fold_idx]

                # Train model
                cv_model = lgb.LGBMClassifier(**lgbm_params)
                cv_model.fit(X_cv_train, y_cv_train)

                # Evaluate on test fold
                y_pred_proba = cv_model.predict_proba(X_cv_test)[:, 1]

                from sklearn.metrics import brier_score_loss
                brier = brier_score_loss(y_cv_test, y_pred_proba)
                cv_brier_fold_scores.append(brier)

            # Train final model on full train set
            final_model = lgb.LGBMClassifier(**lgbm_params)
            final_model.fit(X_train_full, y_train_full)

            # Predict on calibration set
            y_calib_pred_proba = final_model.predict_proba(X_calib)[:, 1]

            # Fit calibrator on calibration set
            from quant_tick.lib.ml import calibrate_per_horizon
            calibrator, calib_method = calibrate_per_horizon(y_calib, y_calib_pred_proba)
            logger.info(f"{model_key}: fitted {calib_method} calibrator on {len(y_calib)} samples")

            # Predict on test set
            y_test_pred_proba_raw = final_model.predict_proba(X_test)[:, 1]

            # Apply calibration to test predictions
            if calibrator is not None:
                if calib_method == "isotonic":
                    y_test_pred_proba = calibrator.transform(y_test_pred_proba_raw)
                elif calib_method == "platt":
                    y_test_pred_proba = calibrator.predict_proba(
                        y_test_pred_proba_raw.reshape(-1, 1)
                    )[:, 1]
                else:
                    y_test_pred_proba = y_test_pred_proba_raw
            else:
                y_test_pred_proba = y_test_pred_proba_raw

            # Compute test metrics AFTER calibration
            brier_test_calibrated = brier_score_loss(y_test, y_test_pred_proba)
            brier_test_raw = brier_score_loss(y_test, y_test_pred_proba_raw)

            # Calibration metrics
            empirical_rate = float(np.mean(y_test))
            predicted_rate_calibrated = float(np.mean(y_test_pred_proba))
            predicted_rate_raw = float(np.mean(y_test_pred_proba_raw))
            calib_gap_calibrated = abs(predicted_rate_calibrated - empirical_rate)
            calib_gap_raw = abs(predicted_rate_raw - empirical_rate)

            # Store models, calibrators, and metrics
            models_dict[model_key] = final_model
            calibrators_dict[model_key] = calibrator
            calibration_methods_dict[model_key] = calib_method

            cv_brier_scores[model_key] = float(np.mean(cv_brier_fold_scores))
            holdout_brier_scores[model_key] = float(brier_test_calibrated)  # CALIBRATED
            holdout_calibration_gaps[model_key] = calib_gap_calibrated  # CALIBRATED

            # Log detailed metrics (both raw and calibrated for comparison)
            logger.info(
                f"{model_key}: CV Brier={cv_brier_scores[model_key]:.4f}, "
                f"Test Brier (raw)={brier_test_raw:.4f}, "
                f"Test Brier (calibrated)={brier_test_calibrated:.4f}, "
                f"Calib gap (raw)={calib_gap_raw:.4f}, "
                f"Calib gap (calibrated)={calib_gap_calibrated:.4f}"
            )

    # Aggregate metrics by side
    lower_cv_briers = [cv_brier_scores[f"lower_h{h}"] for h in decision_horizons]
    upper_cv_briers = [cv_brier_scores[f"upper_h{h}"] for h in decision_horizons]
    lower_holdout_briers = [holdout_brier_scores[f"lower_h{h}"] for h in decision_horizons]
    upper_holdout_briers = [holdout_brier_scores[f"upper_h{h}"] for h in decision_horizons]

    cv_metrics = {
        "cv_brier_scores": cv_brier_scores,  # Keep per-model for debugging
        "avg_brier_lower": float(np.mean(lower_cv_briers)),
        "avg_brier_upper": float(np.mean(upper_cv_briers)),
    }

    # Build per-horizon dicts for config storage
    per_horizon_brier_lower = {h: holdout_brier_scores[f"lower_h{h}"] for h in decision_horizons}
    per_horizon_brier_upper = {h: holdout_brier_scores[f"upper_h{h}"] for h in decision_horizons}

    holdout_metrics = {
        "holdout_brier_scores": holdout_brier_scores,  # Keep per-model
        "calibration_gaps": holdout_calibration_gaps,  # Keep per-model
        "avg_brier_lower": float(np.mean(lower_holdout_briers)),
        "avg_brier_upper": float(np.mean(upper_holdout_briers)),
        "per_horizon_brier_lower": per_horizon_brier_lower,
        "per_horizon_brier_upper": per_horizon_brier_upper,
        "base_rates": base_rates,  # Training base rates for drift monitoring
    }

    # Store calibrators and methods with models (for later serialization)
    for key, calibrator in calibrators_dict.items():
        if calibrator is not None:
            models_dict[key].calibrator_ = calibrator
        models_dict[key].calibration_method_ = calibration_methods_dict.get(key, "none")

    return models_dict, feature_cols, cv_metrics, holdout_metrics


def train_models(
    config: MLConfig,
    n_splits: int = 5,
    embargo_bars: int = 96,
    n_estimators: int = 300,
    max_depth: int = 6,
    min_samples_leaf: int = 50,
    learning_rate: float = 0.05,
    subsample: float = 0.75,
    holdout_pct: float = 0.2,
) -> bool:
    """Train per-horizon direct classifiers.

    Args:
        config: MLConfig to train models for
        n_splits: Number of CV folds
        embargo_bars: Embargo period in bars
        n_estimators: Number of boosting iterations
        max_depth: Max tree depth
        min_samples_leaf: Minimum samples per leaf
        learning_rate: Boosting learning rate
        subsample: Fraction of samples for each tree
        holdout_pct: Fraction of bars to reserve as holdout
    """
    # Get latest feature data
    feature_data = MLFeatureData.objects.filter(
        candle=config.candle
    ).order_by("-timestamp_to").first()

    if not feature_data or not feature_data.has_data_frame("file_data"):
        logger.error(f"{config}: no feature data available")
        return False

    # Validate schema matches config
    is_valid, error_msg = feature_data.validate_schema(config)
    if not is_valid:
        logger.error(
            f"{config}: schema validation failed - {error_msg}. "
            f"Run ml_labels command to regenerate features with correct schema."
        )
        return False

    logger.info(f"{config}: schema validation passed")

    df = feature_data.get_data_frame("file_data")
    logger.info(f"{config}: loaded {len(df)} rows from feature data")

    # Get decision horizons from config or use defaults
    decision_horizons = config.json_data.get("decision_horizons", [60, 120, 180])

    # Train models using core function
    models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
        df=df,
        decision_horizons=decision_horizons,
        n_splits=n_splits,
        embargo_bars=embargo_bars,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        subsample=subsample,
        holdout_pct=holdout_pct,
    )

    logger.info(
        f"{config}: CV Brier - lower={cv_metrics.get('avg_brier_lower', 0):.4f}, "
        f"upper={cv_metrics.get('avg_brier_upper', 0):.4f}"
    )
    logger.info(
        f"{config}: Holdout Brier - lower={holdout_metrics.get('avg_brier_lower', 0):.4f}, "
        f"upper={holdout_metrics.get('avg_brier_upper', 0):.4f}"
    )

    # Compute feature hash for drift detection
    feature_hash = compute_feature_hash(feature_cols)

    # Store model metadata in config
    config.json_data["model_kind"] = "per_horizon_lgbm"
    config.json_data["model_version"] = "2.0"
    if "decision_horizons" not in config.json_data:
        config.json_data["decision_horizons"] = decision_horizons

    # Store holdout metrics in config
    if "holdout_metrics" not in config.json_data:
        config.json_data["holdout_metrics"] = {}
    config.json_data["holdout_metrics"]["lower"] = {
        "brier_cv": cv_metrics.get("avg_brier_lower", 0.0),
        "brier_holdout": holdout_metrics.get("avg_brier_lower", 0.0),
        "per_horizon_brier": holdout_metrics.get("per_horizon_brier_lower", {}),
    }
    config.json_data["holdout_metrics"]["upper"] = {
        "brier_cv": cv_metrics.get("avg_brier_upper", 0.0),
        "brier_holdout": holdout_metrics.get("avg_brier_upper", 0.0),
        "per_horizon_brier": holdout_metrics.get("per_horizon_brier_upper", {}),
    }
    config.save(update_fields=["json_data"])

    # Save per-horizon models
    for h in decision_horizons:
        lower_key = f"lower_h{h}"
        upper_key = f"upper_h{h}"

        if lower_key in models_dict:
            lower_model = models_dict[lower_key]
            lower_brier = holdout_metrics.get("per_horizon_brier_lower", {}).get(h, 0.0)
            calibrator = getattr(lower_model, "calibrator_", None)
            calib_method = getattr(lower_model, "calibration_method_", "none")
            _save_per_horizon_artifact(
                config, lower_model, lower_key, feature_cols, lower_brier, feature_hash, calibrator, h, calib_method
            )

        if upper_key in models_dict:
            upper_model = models_dict[upper_key]
            upper_brier = holdout_metrics.get("per_horizon_brier_upper", {}).get(h, 0.0)
            calibrator = getattr(upper_model, "calibrator_", None)
            calib_method = getattr(upper_model, "calibration_method_", "none")
            _save_per_horizon_artifact(
                config, upper_model, upper_key, feature_cols, upper_brier, feature_hash, calibrator, h, calib_method
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


def _save_per_horizon_artifact(
    config: MLConfig,
    model: Any,
    model_type: str,
    feature_cols: list[str],
    brier_score: float,
    feature_hash: str,
    calibrator: Any,
    horizon: int,
    calibration_method: str = "none",
    base_rate: float | None = None,
) -> None:
    """Save per-horizon model artifact with calibrator and base rate."""
    # Delete existing artifacts of same type
    MLArtifact.objects.filter(
        ml_config=config,
        model_type=model_type,
    ).delete()

    # Serialize model
    buf = BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)

    # Serialize calibrator if exists
    calibrator_bytes = None
    if calibrator is not None:
        calibrator_bytes = pickle.dumps(calibrator)

    # Prepare metadata
    json_data = {}
    if base_rate is not None:
        json_data["base_rate"] = base_rate

    artifact = MLArtifact(
        ml_config=config,
        model_type=model_type,
        brier_score=brier_score,
        feature_columns=feature_cols,
        calibrator=calibrator_bytes,
        horizon=horizon,
        calibration_method=calibration_method,
        json_data=json_data,
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

    cal_status = "with calibrator" if calibrator is not None else "no calibrator"
    logger.info(f"{config}: saved {model_type} model artifact ({cal_status}, feature_hash={feature_hash})")
