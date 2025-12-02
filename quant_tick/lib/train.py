import hashlib
import logging
import pickle
from io import BytesIO
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from django.core.files.base import ContentFile
from sklearn.metrics import brier_score_loss

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


def _cv_brier_for_params(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    timestamp_idx_train: np.ndarray,
    horizon: int,
    n_splits: int,
    embargo_bars: int,
    lgbm_params: dict[str, Any],
) -> float:
    """Compute mean CV Brier score using PurgedKFold.

    Args:
        X_train_full: Training features
        y_train_full: Training labels
        timestamp_idx_train: Timestamp indices for purging
        horizon: Decision horizon for event_end_idx calculation
        n_splits: Number of CV folds
        embargo_bars: Embargo period in bars
        lgbm_params: LightGBM parameters dict

    Returns:
        Mean Brier score across CV folds

    Raises:
        ValueError: If PurgedKFold yields zero folds
    """
    cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
    event_end_idx = timestamp_idx_train + horizon + 1

    fold_list = list(cv.split(
        X_train_full,
        event_end_idx=event_end_idx,
        timestamp_idx=timestamp_idx_train,
    ))

    if len(fold_list) == 0:
        raise ValueError(
            f"PurgedKFold yielded zero folds with n_splits={n_splits}, "
            f"embargo_bars={embargo_bars}. Purging/embargo too aggressive for dataset size."
        )

    brier_scores = []
    for train_idx, test_idx in fold_list:
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_train_full[train_idx], y_train_full[train_idx])
        y_pred = model.predict_proba(X_train_full[test_idx])[:, 1]
        brier_scores.append(brier_score_loss(y_train_full[test_idx], y_pred))

    return float(np.mean(brier_scores))


def _tune_lgbm_hyperparams_optuna(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    timestamp_idx_train: np.ndarray,
    horizon: int,
    side: str,
    n_splits: int,
    embargo_bars: int,
    base_lgbm_params: dict[str, Any],
    n_trials: int = 20,
) -> dict[str, Any]:
    """Tune LightGBM hyperparameters using Optuna.

    Search space:
    - learning_rate: log-uniform [0.01, 0.2]
    - max_depth: int [3, 8]
    - n_estimators: int [100, 400]

    Fixed (not tuned): subsample, min_child_samples, objective, etc.

    Args:
        X_train_full: Training features
        y_train_full: Training labels
        timestamp_idx_train: Timestamp indices for purging
        horizon: Decision horizon
        side: "lower" or "upper" (for logging)
        n_splits: Number of CV folds
        embargo_bars: Embargo period in bars
        base_lgbm_params: Base LightGBM params (non-tuned params copied from here)
        n_trials: Number of Optuna trials (default: 20)

    Returns:
        Best LightGBM params (base params + tuned values)
    """
    def objective(trial: Any) -> float:
        trial_params = base_lgbm_params.copy()
        trial_params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        })

        return _cv_brier_for_params(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            timestamp_idx_train=timestamp_idx_train,
            horizon=horizon,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            lgbm_params=trial_params,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Optuna tuning {side}_h{horizon} ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        f"{side}_h{horizon}: best Brier={study.best_trial.value:.4f}, "
        f"params={study.best_trial.params}"
    )

    best_params = base_lgbm_params.copy()
    best_params.update(study.best_trial.params)
    return best_params


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
    optuna_n_trials: int = 20,
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

    This is a per-horizon logistic regression with trees.

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
        optuna_n_trials: Number of Optuna trials per model (default: 20).
            Set to 0 to disable tuning and use fixed params.

    Returns:
        Tuple of (models_dict, feature_cols, cv_metrics, holdout_metrics)
        - models_dict: {model_key: trained_model} with calibrator attached as .calibrator_
        - feature_cols: List of feature names used
        - cv_metrics: Cross-validation Brier scores
        - holdout_metrics: Holdout Brier scores and calibration gaps
    """
    from quant_tick.lib.schema import MLSchema

    # Validate bar_idx and config_id invariants
    is_valid, error_msg = MLSchema.validate_bar_config_invariants(df)
    if not is_valid:
        raise ValueError(f"Bar/config invariants validation failed: {error_msg}")

    # Infer widths and asymmetries from data if available
    widths = None
    asymmetries = None
    if "width" in df.columns and "asymmetry" in df.columns:
        widths = sorted(df["width"].unique())
        asymmetries = sorted(df["asymmetry"].unique())

        # Validate complete grid structure at input
        is_valid, error_msg = MLSchema.validate_bar_config_structure(df, widths, asymmetries)
        if not is_valid:
            raise ValueError(f"Input grid structure validation failed: {error_msg}")

    # Identify feature columns using MLSchema
    feature_cols = MLSchema.get_training_features(df.columns.tolist(), decision_horizons)

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

    # Log detailed purging math
    logger.info(
        f"Purging math:\n"
        f"  Total bars: {n_bars}\n"
        f"  Max horizon: {max_horizon}\n"
        f"  Initial split: train={train_bars_initial}, calib={calib_bars}, test={test_bars}\n"
        f"  Purge amounts: train loses {train_bars_initial - train_bars_purged} bars, "
        f"calib loses {calib_bars - calib_bars_purged} bars\n"
        f"  After purging: train={train_bars_purged}, calib={calib_bars_purged}, test={test_bars}\n"
        f"  Bar ranges: train=[0, {train_bars_purged}), "
        f"calib=[{first_calib_bar}, {first_calib_bar + calib_bars_purged}), "
        f"test=[{first_test_bar}, {first_test_bar + test_bars})"
    )

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

    # Validate purged training set has complete grids
    if widths is not None and asymmetries is not None:
        train_df = df.iloc[train_idx]
        is_valid, error_msg = MLSchema.validate_bar_config_structure(train_df, widths, asymmetries)
        if not is_valid:
            raise ValueError(f"Purged training set grid validation failed: {error_msg}")

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
    optuna_best_params = {}  # Store best Optuna params for all models

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

            # Determine params for this model (with or without Optuna)
            if optuna_n_trials > 0:
                # Run Optuna tuning for this (side, horizon) model
                tuned_params = _tune_lgbm_hyperparams_optuna(
                    X_train_full=X_train_full,
                    y_train_full=y_train_full,
                    timestamp_idx_train=timestamp_idx_train,
                    horizon=h,
                    side=side,
                    n_splits=n_splits,
                    embargo_bars=embargo_bars,
                    base_lgbm_params=lgbm_params,
                    n_trials=optuna_n_trials,
                )
                # Store best params for this model
                optuna_best_params[model_key] = {
                    k: tuned_params[k]
                    for k in ["learning_rate", "max_depth", "n_estimators"]
                }
                model_params = tuned_params
            else:
                # Use fixed params (no Optuna)
                model_params = lgbm_params

            # Cross-validation with final params
            # (This recomputes CV with tuned params - necessary for honest CV metric)
            mean_cv_brier = _cv_brier_for_params(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                timestamp_idx_train=timestamp_idx_train,
                horizon=h,
                n_splits=n_splits,
                embargo_bars=embargo_bars,
                lgbm_params=model_params,
            )
            cv_brier_scores[model_key] = mean_cv_brier

            # Train final model on full train set with chosen params
            final_model = lgb.LGBMClassifier(**model_params)
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

            holdout_brier_scores[model_key] = float(brier_test_calibrated)  # CALIBRATED
            holdout_calibration_gaps[model_key] = calib_gap_calibrated  # CALIBRATED

            # Log detailed metrics (both raw and calibrated for comparison)
            optuna_status = "(Optuna tuned)" if optuna_n_trials > 0 else "(fixed params)"
            logger.info(
                f"{model_key} {optuna_status}: CV Brier={mean_cv_brier:.4f}, "
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
        "optuna_best_params": optuna_best_params,  # Best Optuna params per model
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

    # Get Optuna config (default: 20 trials)
    optuna_n_trials = config.json_data.get("optuna_n_trials", 20)

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
        optuna_n_trials=optuna_n_trials,
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

    # Store Optuna best params if tuning was enabled
    if optuna_n_trials > 0:
        optuna_best_params = cv_metrics.get("optuna_best_params", {})
        if optuna_best_params:
            config.json_data["optuna_best_params"] = optuna_best_params
            logger.info(f"{config}: stored Optuna best params for {len(optuna_best_params)} models")

    config.save(update_fields=["json_data"])

    # Save per-horizon models
    base_rates = holdout_metrics.get("base_rates", {})
    for h in decision_horizons:
        lower_key = f"lower_h{h}"
        upper_key = f"upper_h{h}"

        if lower_key in models_dict:
            lower_model = models_dict[lower_key]
            lower_brier = holdout_metrics.get("per_horizon_brier_lower", {}).get(h, 0.0)
            calibrator = getattr(lower_model, "calibrator_", None)
            calib_method = getattr(lower_model, "calibration_method_", "none")
            lower_base_rate = base_rates.get(lower_key)
            _save_per_horizon_artifact(
                config, lower_model, lower_key, feature_cols, lower_brier, feature_hash, calibrator, h, calib_method, lower_base_rate
            )

        if upper_key in models_dict:
            upper_model = models_dict[upper_key]
            upper_brier = holdout_metrics.get("per_horizon_brier_upper", {}).get(h, 0.0)
            calibrator = getattr(upper_model, "calibrator_", None)
            calib_method = getattr(upper_model, "calibration_method_", "none")
            upper_base_rate = base_rates.get(upper_key)
            _save_per_horizon_artifact(
                config, upper_model, upper_key, feature_cols, upper_brier, feature_hash, calibrator, h, calib_method, upper_base_rate
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
