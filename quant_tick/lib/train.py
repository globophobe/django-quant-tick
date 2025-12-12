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
    apply_calibration,
    calibrate_per_horizon,
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
    model_params: dict[str, Any],
) -> float:
    """Compute mean CV Brier score using PurgedKFold.

    Args:
        X_train_full: Training features
        y_train_full: Training labels
        timestamp_idx_train: Timestamp indices for purging
        horizon: Decision horizon for event_end_idx calculation
        n_splits: Number of CV folds
        embargo_bars: Embargo_bars period in bars
        model_params: Model parameters dict

    Returns:
        Mean Brier score across CV folds

    Raises:
        ValueError: If PurgedKFold yields zero folds
    """
    cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
    event_end_idx = timestamp_idx_train + horizon + 1

    fold_list = list(
        cv.split(
            X_train_full,
            event_end_idx=event_end_idx,
            timestamp_idx=timestamp_idx_train,
        )
    )

    if len(fold_list) == 0:
        raise ValueError(
            f"PurgedKFold yielded zero folds with n_splits={n_splits}, "
            f"embargo_bars={embargo_bars}. Purging/embargo too aggressive for dataset size."
        )

    brier_scores = []
    for train_idx, test_idx in fold_list:
        model = lgb.LGBMClassifier(**model_params)
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
        trial_params.update(
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            }
        )

        return _cv_brier_for_params(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            timestamp_idx_train=timestamp_idx_train,
            horizon=horizon,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            model_params=trial_params,
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


def train_core(
    df: pd.DataFrame,
    max_horizon: int,
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
    """Train survival models for range touch prediction.

    Trains 2 models (lower, upper) for all time steps up to max_horizon,
    using discrete-time survival analysis for competing risks.

    Args:
        df: Hazard-labeled DataFrame (n_bars × n_configs × max_horizon rows)
        max_horizon: Maximum time steps (T)
        n_splits: PurgedKFold splits
        embargo_bars: Embargo buffer for purging
        n_estimators: LightGBM n_estimators
        max_depth: LightGBM max_depth
        min_samples_leaf: LightGBM min_child_samples
        learning_rate: LightGBM learning_rate
        subsample: LightGBM subsample
        holdout_pct: Final test set fraction
        calibration_pct: Calibration set fraction
        optuna_n_trials: Hyperparameter tuning trials (0=disable)

    Returns:
        Tuple of (models_dict, feature_cols, cv_metrics, holdout_metrics)
        - models_dict: {"lower": model, "upper": model}
            Each model has calibrator_ and calibration_method_ attributes
        - feature_cols: List of feature names (includes "k")
        - cv_metrics: Cross-validation Brier scores
        - holdout_metrics: Holdout performance and base rates
    """
    from quant_tick.lib.schema import MLSchema

    logger.info("Starting model training")

    feature_cols = MLSchema.get_training_features(df.columns.tolist())
    X, feature_cols = prepare_features(df, feature_cols)

    logger.info(f"Training features: {len(feature_cols)} columns (includes k)")

    unique_bar_indices = df["bar_idx"].unique()
    n_bars = len(unique_bar_indices)

    rows_per_bar = len(df[df["bar_idx"] == unique_bar_indices[0]])
    n_configs = rows_per_bar // max_horizon

    if rows_per_bar != n_configs * max_horizon:
        raise ValueError(
            f"Invalid row count per bar: {rows_per_bar} != {n_configs} × {max_horizon}"
        )

    logger.info(
        f"Grid structure: {n_bars} bars × {n_configs} configs × {max_horizon} time steps "
        f"= {len(df)} rows"
    )

    test_bars = int(n_bars * holdout_pct)
    calib_bars = int(n_bars * calibration_pct)
    train_bars_initial = n_bars - test_bars - calib_bars

    train_bars_purged = max(0, train_bars_initial - (max_horizon + 1))
    calib_bars_purged = max(0, calib_bars - (max_horizon + 1))

    if train_bars_purged < 100:
        raise ValueError(
            f"Insufficient training data after purging: {train_bars_purged} bars"
        )

    rows_per_bar_total = n_configs * max_horizon

    train_idx = np.arange(train_bars_purged * rows_per_bar_total)

    first_calib_bar = train_bars_purged + max_horizon + 1
    calib_idx = np.arange(
        first_calib_bar * rows_per_bar_total,
        (first_calib_bar + calib_bars_purged) * rows_per_bar_total,
    )

    first_test_bar = first_calib_bar + calib_bars_purged + max_horizon + 1
    test_idx = np.arange(first_test_bar * rows_per_bar_total, len(df))

    logger.info(
        f"Train bars: {train_bars_purged}, "
        f"Calib bars: {calib_bars_purged}, "
        f"Test bars: {n_bars - first_test_bar}"
    )

    X_train_full = X[train_idx]
    X_calib = X[calib_idx]
    X_test = X[test_idx]

    timestamp_idx_train = df.iloc[train_idx]["bar_idx"].values

    models_dict = {}
    cv_metrics = {"cv_brier_scores": {}, "optuna_best_params": {}}
    holdout_metrics = {"holdout_brier_scores": {}, "base_rates": {}}

    base_lgbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "min_child_samples": min_samples_leaf,
        "subsample": subsample,
        "subsample_freq": 1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1,
    }

    for side in ["lower", "upper"]:
        label_col = f"hazard_{side}"
        model_key = side

        logger.info(f"\n{'='*60}")
        logger.info(f"Training {side} model")
        logger.info(f"{'='*60}")

        y_train_full = df.iloc[train_idx][label_col].values
        y_calib = df.iloc[calib_idx][label_col].values
        y_test = df.iloc[test_idx][label_col].values

        train_base_rate = float(y_train_full.mean())
        logger.info(f"Training set base rate (P(hazard=1)): {train_base_rate:.4f}")

        if optuna_n_trials > 0:
            logger.info(f"Starting Optuna tuning: {optuna_n_trials} trials")

            tuned_params = _tune_lgbm_hyperparams_optuna(
                X_train_full,
                y_train_full,
                timestamp_idx_train,
                horizon=max_horizon,
                side=side,
                n_trials=optuna_n_trials,
                n_splits=n_splits,
                embargo_bars=embargo_bars,
                base_lgbm_params=base_lgbm_params,
            )

            model_params = tuned_params
            cv_metrics["optuna_best_params"][model_key] = tuned_params
        else:
            model_params = base_lgbm_params.copy()
            model_params.update(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                }
            )

        logger.info("Running cross-validation with PurgedKFold")

        mean_cv_brier = _cv_brier_for_params(
            X_train_full,
            y_train_full,
            timestamp_idx_train,
            horizon=max_horizon,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            model_params=model_params,
        )

        cv_metrics["cv_brier_scores"][model_key] = mean_cv_brier
        logger.info(f"CV Brier score: {mean_cv_brier:.4f}")

        logger.info("Training final model on full training set")

        final_model = lgb.LGBMClassifier(**model_params)
        final_model.fit(
            X_train_full,
            y_train_full,
            eval_set=[(X_calib, y_calib)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        logger.info(f"Training complete: {final_model.n_estimators_} trees")

        logger.info("Calibrating on calibration set")

        y_calib_pred_proba = final_model.predict_proba(X_calib)[:, 1]
        calibrator, calib_method = calibrate_per_horizon(y_calib, y_calib_pred_proba)

        final_model.calibrator_ = calibrator
        final_model.calibration_method_ = calib_method

        logger.info(f"Calibration: method={calib_method}")

        logger.info("Evaluating on holdout test set")

        y_test_pred_proba_raw = final_model.predict_proba(X_test)[:, 1]
        y_test_pred_proba = apply_calibration(
            y_test_pred_proba_raw, calibrator, calib_method
        )

        holdout_brier = brier_score_loss(y_test, y_test_pred_proba)
        holdout_metrics["holdout_brier_scores"][model_key] = holdout_brier
        holdout_metrics["base_rates"][model_key] = train_base_rate

        logger.info(f"Holdout Brier score: {holdout_brier:.4f}")

        models_dict[model_key] = final_model

    cv_metrics["avg_brier"] = float(
        np.mean(list(cv_metrics["cv_brier_scores"].values()))
    )
    holdout_metrics["avg_brier"] = float(
        np.mean(list(holdout_metrics["holdout_brier_scores"].values()))
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training complete")
    logger.info(f"Average CV Brier: {cv_metrics['avg_brier']:.4f}")
    logger.info(f"Average holdout Brier: {holdout_metrics['avg_brier']:.4f}")
    logger.info("=" * 60 + "\n")

    return models_dict, feature_cols, cv_metrics, holdout_metrics


def train_models(config: MLConfig, **override_params) -> bool:
    """Train survival models for MLConfig.

    Args:
        config: MLConfig with associated MLFeatureData
        **override_params: Override any training parameter for this run
            (n_splits, embargo_bars, n_estimators, max_depth,
             min_samples_leaf, learning_rate, subsample, holdout_pct, calibration_pct)

    Returns:
        True if training succeeded, False otherwise

    Raises:
        ValueError: If no feature data or schema validation fails
    """
    logger.info(f"Training models for config: {config.code_name}")

    # Get training params from config and merge with overrides
    training_params = config.get_training_params()
    training_params.update(override_params)

    # Extract individual params
    n_splits = training_params["n_splits"]
    embargo_bars = training_params["embargo_bars"]
    n_estimators = training_params["n_estimators"]
    max_depth = training_params["max_depth"]
    min_samples_leaf = training_params["min_samples_leaf"]
    learning_rate = training_params["learning_rate"]
    subsample = training_params["subsample"]
    holdout_pct = training_params["holdout_pct"]
    calibration_pct = training_params["calibration_pct"]

    # Get optuna trials from config
    optuna_n_trials = config.get_optuna_n_trials()

    feature_data = (
        MLFeatureData.objects.filter(candle=config.candle)
        .order_by("-timestamp_to")
        .first()
    )

    if not feature_data or not feature_data.has_data_frame("file_data"):
        logger.error(f"{config}: no feature data available")
        return False

    is_valid, error_msg = feature_data.validate_schema(config)
    if not is_valid:
        logger.error(
            f"{config}: schema validation failed - {error_msg}. "
            f"Run ml_labels to regenerate features."
        )
        return False

    schema_type = feature_data.json_data.get("schema_type", "per_horizon")
    if schema_type != "hazard":
        logger.error(
            f"{config}: schema_type='{schema_type}', expected 'hazard'. "
            f"Run ml_labels to regenerate features."
        )
        return False

    df = feature_data.get_data_frame("file_data")
    max_horizon = feature_data.json_data.get("max_horizon", config.horizon_bars)

    logger.info(
        f"{config}: loaded {len(df)} rows, "
        f"max_horizon={max_horizon}, "
        f"schema_hash={feature_data.schema_hash[:8]}"
    )

    models_dict, feature_cols, cv_metrics, holdout_metrics = train_core(
        df,
        max_horizon=max_horizon,
        n_splits=n_splits,
        embargo_bars=embargo_bars,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        subsample=subsample,
        holdout_pct=holdout_pct,
        calibration_pct=calibration_pct,
        optuna_n_trials=optuna_n_trials,
    )

    for side, model in models_dict.items():
        _save_artifact(
            config=config,
            model=model,
            side=side,
            feature_cols=feature_cols,
            brier_score=holdout_metrics["holdout_brier_scores"][side],
            base_rate=holdout_metrics["base_rates"][side],
        )

    config.json_data["model_kind"] = "hazard"
    config.json_data["model_version"] = "3.0"
    config.json_data["max_horizon"] = max_horizon
    config.json_data["cv_metrics"] = cv_metrics
    config.json_data["holdout_metrics"] = holdout_metrics
    config.json_data["feature_hash"] = compute_feature_hash(feature_cols)

    if optuna_n_trials > 0:
        optuna_best_params = cv_metrics.get("optuna_best_params", {})
        if optuna_best_params:
            config.json_data["optuna_best_params"] = optuna_best_params

    config.save(update_fields=["json_data"])

    logger.info(f"{config}: model training complete")
    return True


def _save_artifact(
    config: MLConfig,
    model: Any,
    side: str,
    feature_cols: list[str],
    brier_score: float,
    base_rate: float,
) -> MLArtifact:
    """Save trained model artifact.

    Args:
        config: MLConfig
        model: Trained LightGBM model with calibrator_ attribute
        side: "lower" or "upper"
        feature_cols: Ordered list of feature names
        brier_score: Holdout Brier score
        base_rate: Training set base rate (P(hazard=1))

    Returns:
        Saved MLArtifact
    """
    model_type = f"hazard_{side}"

    MLArtifact.objects.filter(
        ml_config=config,
        model_type=model_type,
    ).delete()

    logger.info(f"Saving {model_type} artifact")

    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    calibrator = getattr(model, "calibrator_", None)
    calibration_method = getattr(model, "calibration_method_", "none")

    artifact = MLArtifact.objects.create(
        ml_config=config,
        model_type=model_type,
        brier_score=brier_score,
        feature_columns=feature_cols,
        calibrator=pickle.dumps(calibrator) if calibrator else b"",
        calibration_method=calibration_method,
        json_data={"base_rate": base_rate},
    )

    artifact.artifact.save(f"{model_type}.joblib", ContentFile(buffer.read()))
    artifact.save()

    logger.info(
        f"Saved {model_type}: "
        f"brier={brier_score:.4f}, "
        f"base_rate={base_rate:.4f}, "
        f"sha256={artifact.sha256[:8]}"
    )

    return artifact
