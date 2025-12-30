import hashlib
import logging
import os
from io import BytesIO
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from django.utils import timezone
from google.cloud import storage
from quant_core.prediction import apply_calibration, prepare_features
from sklearn.metrics import brier_score_loss

from quant_tick.lib.ml import PurgedKFold, calibrate_per_horizon
from quant_tick.models import MLArtifact, MLConfig

logger = logging.getLogger(__name__)


def save_model_bundle_to_gcs(bundle: dict, gcs_path: str) -> None:
    """Save model bundle to GCS.

    Args:
        bundle: {"models": {"lower": model, "upper": model}, "metadata": {...}}
        gcs_path: gs://bucket/path/model.joblib

    Raises:
        ValueError: If gcs_path format is invalid
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")

    parts = gcs_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS path format: {gcs_path}")

    bucket_name, blob_name = parts

    logger.info(f"Uploading model bundle to {gcs_path}")

    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")

    logger.info(f"Upload complete: {gcs_path} ({len(buffer.getvalue()) / 1024 / 1024:.2f} MB)")


def compute_feature_hash(feature_cols: list[str]) -> str:
    """Compute hash of feature columns for drift detection."""
    cols_str = ",".join(sorted(feature_cols))
    return hashlib.sha256(cols_str.encode()).hexdigest()[:16]


def _cv_brier_for_params(
    X_train_full: pd.DataFrame | np.ndarray,
    y_train_full: np.ndarray,
    bar_idx_train: np.ndarray,
    horizon: int,
    n_splits: int,
    embargo_bars: int,
    model_params: dict[str, Any],
) -> float:
    """Compute mean CV Brier score using PurgedKFold.

    Args:
        X_train_full: Training features (DataFrame or numpy array)
        y_train_full: Training labels
        bar_idx_train: Bar indices for purging
        horizon: Decision horizon for event_end_exclusive_idx calculation
        n_splits: Number of CV folds
        embargo_bars: Embargo_bars period in bars
        model_params: Model parameters dict

    Returns:
        Mean Brier score across CV folds

    Raises:
        ValueError: If PurgedKFold yields zero folds
    """
    cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
    # Label at bar i uses offsets 1..H (bars i+1 to i+H), exclusive end = i+H+1
    event_end_exclusive_idx = bar_idx_train + horizon + 1

    fold_list = list(
        cv.split(
            X_train_full,
            event_end_exclusive_idx=event_end_exclusive_idx,
            bar_idx=bar_idx_train,
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
        # Use numpy indexing (works for both DataFrame and ndarray)
        X_fold_train = X_train_full[train_idx] if isinstance(X_train_full, np.ndarray) else X_train_full.iloc[train_idx]
        X_fold_test = X_train_full[test_idx] if isinstance(X_train_full, np.ndarray) else X_train_full.iloc[test_idx]
        model.fit(X_fold_train, y_train_full[train_idx])
        y_pred = model.predict_proba(X_fold_test)[:, 1]
        brier_scores.append(brier_score_loss(y_train_full[test_idx], y_pred))

    return float(np.mean(brier_scores))


def _cv_logloss_for_params(
    X_train_full: pd.DataFrame | np.ndarray,
    y_train_full: np.ndarray,
    bar_idx_train: np.ndarray,
    horizon: int,
    n_splits: int,
    embargo_bars: int,
    model_params: dict[str, Any],
) -> float:
    """Compute mean CV log-loss using PurgedKFold (multiclass).

    Returns mean log-loss across CV folds.
    Raises ValueError if PurgedKFold yields zero folds.

    Handles missing classes in CV folds by forcing labels=[0,1,2] in log_loss.
    """
    from sklearn.metrics import log_loss

    cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
    # Label at bar i uses offsets 1..H (bars i+1 to i+H), exclusive end = i+H+1
    event_end_exclusive_idx = bar_idx_train + horizon + 1

    fold_list = list(cv.split(X_train_full, event_end_exclusive_idx=event_end_exclusive_idx, bar_idx=bar_idx_train))

    if len(fold_list) == 0:
        raise ValueError(f"PurgedKFold yielded zero folds with n_splits={n_splits}, embargo_bars={embargo_bars}")

    logloss_scores = []
    for train_idx, test_idx in fold_list:
        model = lgb.LGBMClassifier(**model_params)
        X_fold_train = X_train_full[train_idx] if isinstance(X_train_full, np.ndarray) else X_train_full.iloc[train_idx]
        X_fold_test = X_train_full[test_idx] if isinstance(X_train_full, np.ndarray) else X_train_full.iloc[test_idx]

        model.fit(X_fold_train, y_train_full[train_idx])
        y_pred_proba_raw = model.predict_proba(X_fold_test)

        # CRITICAL: Remap predict_proba to fixed 3-column layout
        # Time-series CV can have training folds missing classes (e.g., no UP_FIRST early on)
        # Then model.classes_ = [0, 2] and predict_proba returns 2 columns, not 3
        # We must expand to (n_samples, 3) aligned with [0, 1, 2]
        y_pred_proba = np.zeros((len(test_idx), 3))
        for i, cls in enumerate(model.classes_):
            y_pred_proba[:, cls] = y_pred_proba_raw[:, i]

        logloss_scores.append(log_loss(y_train_full[test_idx], y_pred_proba, labels=[0, 1, 2]))

    return float(np.mean(logloss_scores))


def _tune_lgbm_hyperparams_optuna(
    X_train_full: pd.DataFrame | np.ndarray,
    y_train_full: np.ndarray,
    bar_idx_train: np.ndarray,
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
        X_train_full: Training features (DataFrame with feature names)
        y_train_full: Training labels
        bar_idx_train: Bar indices for purging
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
            bar_idx_train=bar_idx_train,
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


def _tune_lgbm_hyperparams_optuna_multiclass(
    X_train_full: pd.DataFrame | np.ndarray,
    y_train_full: np.ndarray,
    bar_idx_train: np.ndarray,
    horizon: int,
    model_name: str,
    n_splits: int,
    embargo_bars: int,
    base_lgbm_params: dict[str, Any],
    n_trials: int = 20,
) -> dict[str, Any]:
    """Tune LightGBM hyperparameters for multiclass using Optuna.

    Search space: learning_rate [0.01, 0.2], max_depth [3, 8], n_estimators [100, 400]
    Returns best params dict.
    """

    def objective(trial: Any) -> float:
        trial_params = base_lgbm_params.copy()
        trial_params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        })

        return _cv_logloss_for_params(
            X_train_full, y_train_full, bar_idx_train,
            horizon, n_splits, embargo_bars, trial_params
        )

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Optuna tuning {model_name} ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"{model_name}: best log-loss={study.best_trial.value:.4f}, params={study.best_trial.params}")

    best_params = base_lgbm_params.copy()
    best_params.update(study.best_trial.params)
    return best_params


def _validate_hazard_grid(
    df: pd.DataFrame, max_horizon: int
) -> tuple[int, int, int]:
    """Validate hazard grid structure and return dimensions.

    Args:
        df: Hazard-labeled DataFrame
        max_horizon: Maximum time steps

    Returns:
        Tuple of (n_bars, n_configs, rows_per_bar)

    Raises:
        ValueError: If grid structure is invalid
    """
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

    return n_bars, n_configs, rows_per_bar


def _compute_time_splits(
    n_bars: int,
    rows_per_bar: int,
    max_horizon: int,
    holdout_pct: float,
    calibration_pct: float,
    min_train_bars_purged: int,
    total_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Compute train/calibration/test indices with purging.

    Args:
        n_bars: Total number of bars
        rows_per_bar: Rows per bar (n_configs × max_horizon)
        max_horizon: Maximum time steps
        holdout_pct: Holdout set fraction
        calibration_pct: Calibration set fraction
        min_train_bars_purged: Minimum training bars after purging
        total_rows: Total rows in dataset

    Returns:
        Tuple of (train_idx, calib_idx, test_idx, train_bars_purged, calib_bars_purged)

    Raises:
        ValueError: If insufficient training data after purging
    """
    test_bars = int(n_bars * holdout_pct)
    calib_bars = int(n_bars * calibration_pct)
    train_bars_initial = n_bars - test_bars - calib_bars

    train_bars_purged = max(0, train_bars_initial - (max_horizon + 1))
    calib_bars_purged = max(0, calib_bars - (max_horizon + 1))

    if train_bars_purged < min_train_bars_purged:
        raise ValueError(
            f"Insufficient training data after purging: {train_bars_purged} bars "
            f"(min required: {min_train_bars_purged})"
        )

    train_idx = np.arange(train_bars_purged * rows_per_bar)

    first_calib_bar = train_bars_purged + max_horizon + 1
    calib_idx = np.arange(
        first_calib_bar * rows_per_bar,
        (first_calib_bar + calib_bars_purged) * rows_per_bar,
    )

    first_test_bar = first_calib_bar + calib_bars_purged + max_horizon + 1
    test_idx = np.arange(first_test_bar * rows_per_bar, total_rows)

    logger.info(
        f"Train bars: {train_bars_purged}, "
        f"Calib bars: {calib_bars_purged}, "
        f"Test bars: {n_bars - first_test_bar}"
    )

    return train_idx, calib_idx, test_idx, train_bars_purged, calib_bars_purged


def _train_side_model(
    side: str,
    df: pd.DataFrame,
    X_train_full: np.ndarray,
    X_calib: np.ndarray,
    X_test: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    bar_idx_train: np.ndarray,
    max_horizon: int,
    n_splits: int,
    embargo_bars: int,
    base_lgbm_params: dict,
    optuna_n_trials: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    label_col: str | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[Any, float, float, float, dict | None]:
    """Train model for one side with CV, optional Optuna tuning, and calibration.

    Args:
        side: Model identifier (e.g., "lower", "upper", "lower_h48")
        df: Full DataFrame with labels
        X_train_full: Training features
        X_calib: Calibration features
        X_test: Test features
        train_idx: Training indices in df
        calib_idx: Calibration indices in df
        test_idx: Test indices in df
        bar_idx_train: Bar indices for purging
        max_horizon: Horizon for event_end_exclusive_idx calculation
        n_splits: PurgedKFold splits
        embargo_bars: Embargo buffer
        base_lgbm_params: Base LightGBM params
        optuna_n_trials: Optuna trials (0=disable)
        n_estimators: LightGBM n_estimators (for non-Optuna)
        max_depth: LightGBM max_depth (for non-Optuna)
        learning_rate: LightGBM learning_rate (for non-Optuna)
        label_col: Label column name (if None, uses f"hazard_{side}")

    Returns:
        Tuple of (model, cv_brier, holdout_brier, base_rate, optuna_best_params_or_none)
    """
    if label_col is None:
        label_col = f"hazard_{side}"

    y_train_full = df.iloc[train_idx][label_col].values
    y_calib = df.iloc[calib_idx][label_col].values
    y_test = df.iloc[test_idx][label_col].values

    train_base_rate = float(y_train_full.mean())
    logger.info(f"Training set base rate (P(touched=1)): {train_base_rate:.4f}")

    # Convert to DataFrames for consistent feature names
    X_train_df = pd.DataFrame(X_train_full, columns=feature_cols)
    X_calib_df = pd.DataFrame(X_calib, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    optuna_best_params = None
    if optuna_n_trials > 0:
        logger.info(f"Starting Optuna tuning: {optuna_n_trials} trials")

        tuned_params = _tune_lgbm_hyperparams_optuna(
            X_train_df,
            y_train_full,
            bar_idx_train,
            horizon=max_horizon,
            side=side,
            n_trials=optuna_n_trials,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            base_lgbm_params=base_lgbm_params,
        )

        model_params = tuned_params
        optuna_best_params = tuned_params
    else:
        model_params = base_lgbm_params.copy()
        model_params.update(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            }
        )

    if n_splits <= 1:
        logger.info("Skipping CV (n_splits <= 1)")
        mean_cv_brier = np.nan
    else:
        logger.info("Running cross-validation with PurgedKFold")
        mean_cv_brier = _cv_brier_for_params(
            X_train_df,
            y_train_full,
            bar_idx_train,
            horizon=max_horizon,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            model_params=model_params,
        )
        logger.info(f"CV Brier score: {mean_cv_brier:.4f}")

    logger.info("Training final model on full training set")

    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(
        X_train_df,
        y_train_full,
        eval_set=[(X_calib_df, y_calib)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    logger.info(f"Training complete: {final_model.n_estimators_} trees")

    logger.info("Calibrating on calibration set")

    y_calib_pred_proba = final_model.predict_proba(X_calib_df)[:, 1]
    calibrator, calib_method = calibrate_per_horizon(y_calib, y_calib_pred_proba)

    final_model.calibrator_ = calibrator
    final_model.calibration_method_ = calib_method

    logger.info(f"Calibration: method={calib_method}")

    logger.info("Evaluating on holdout test set")

    y_test_pred_proba_raw = final_model.predict_proba(X_test_df)[:, 1]
    y_test_pred_proba = apply_calibration(
        y_test_pred_proba_raw, calibrator, calib_method
    )

    holdout_brier = brier_score_loss(y_test, y_test_pred_proba)

    logger.info(f"Holdout Brier score: {holdout_brier:.4f}")

    return final_model, mean_cv_brier, holdout_brier, train_base_rate, optuna_best_params


def _remap_proba_via_classes(y_pred_proba: np.ndarray, model: Any) -> np.ndarray:
    """Remap predict_proba to canonical [0,1,2] order using model.classes_.

    LightGBM returns probs in model.classes_ order, not [0,1,2] order.
    Must match inference behavior (quant_core/prediction.py:123-131).

    Args:
        y_pred_proba: Raw predictions from model.predict_proba() [n_samples, n_classes]
        model: Trained model with classes_ attribute

    Returns:
        Remapped predictions in canonical [0, 1, 2] order
    """
    if not hasattr(model, 'classes_'):
        return y_pred_proba

    n_samples = y_pred_proba.shape[0]
    probs_remapped = np.zeros((n_samples, 3))
    for i, cls in enumerate(model.classes_):
        probs_remapped[:, cls] = y_pred_proba[:, i]
    return probs_remapped


def _train_multiclass_model(
    model_name: str,
    df: pd.DataFrame,
    X_train_full: np.ndarray,
    X_calib: np.ndarray,
    X_test: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    bar_idx_train: np.ndarray,
    max_horizon: int,
    n_splits: int,
    embargo_bars: int,
    base_lgbm_params: dict,
    optuna_n_trials: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    label_col: str,
    feature_cols: list[str],
) -> tuple[Any, float, float, dict, dict | None]:
    """Train multiclass model for competing-risks.

    Returns:
        Tuple of (model, cv_logloss, holdout_logloss, class_distribution, optuna_params)
    """
    from sklearn.metrics import log_loss

    y_train_full = df.iloc[train_idx][label_col].values
    y_calib = df.iloc[calib_idx][label_col].values
    y_test = df.iloc[test_idx][label_col].values

    # Log class distribution
    class_counts = np.bincount(y_train_full, minlength=3)
    class_dist = {
        "UP_FIRST": float(class_counts[0] / len(y_train_full)),
        "DOWN_FIRST": float(class_counts[1] / len(y_train_full)),
        "TIMEOUT": float(class_counts[2] / len(y_train_full)),
    }
    logger.info(f"Training set class distribution: {class_dist}")

    # Convert to DataFrames for consistent feature names
    X_train_df = pd.DataFrame(X_train_full, columns=feature_cols)
    X_calib_df = pd.DataFrame(X_calib, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    # Use multiclass-specific helpers
    optuna_best_params = None
    if optuna_n_trials > 0:
        logger.info(f"Starting Optuna tuning: {optuna_n_trials} trials")

        tuned_params = _tune_lgbm_hyperparams_optuna_multiclass(
            X_train_df, y_train_full, bar_idx_train,
            horizon=max_horizon, model_name=model_name,
            n_trials=optuna_n_trials, n_splits=n_splits,
            embargo_bars=embargo_bars, base_lgbm_params=base_lgbm_params,
        )

        model_params = tuned_params
        optuna_best_params = tuned_params
    else:
        model_params = base_lgbm_params.copy()
        model_params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        })

    # Run CV if requested
    if n_splits <= 1:
        logger.info("Skipping CV (n_splits <= 1)")
        mean_cv_logloss = np.nan
    else:
        logger.info("Running cross-validation with PurgedKFold")
        mean_cv_logloss = _cv_logloss_for_params(
            X_train_df, y_train_full, bar_idx_train,
            horizon=max_horizon, n_splits=n_splits,
            embargo_bars=embargo_bars, model_params=model_params,
        )
        logger.info(f"CV log-loss: {mean_cv_logloss:.4f}")

    # Train final model
    logger.info("Training final model on full training set")
    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(
        X_train_df, y_train_full,
        eval_set=[(X_calib_df, y_calib)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    logger.info(f"Training complete: {final_model.n_estimators_} trees")

    # Calibrate using One-vs-Rest approach
    logger.info("Calibrating on calibration set (One-vs-Rest)")

    y_calib_pred_proba_raw = final_model.predict_proba(X_calib_df)
    # CRITICAL: Remap to canonical order before calibration
    y_calib_pred_proba = _remap_proba_via_classes(y_calib_pred_proba_raw, final_model)

    from quant_tick.lib.ml import calibrate_multiclass_ovr

    calibrators, methods = calibrate_multiclass_ovr(y_calib, y_calib_pred_proba)

    # Store as custom attributes (will be pickled with model)
    final_model.calibrator_ = calibrators          # {0: calibrator, 1: calibrator, 2: None}
    final_model.calibration_methods_ = methods     # {0: "isotonic", 1: "platt", 2: "none"}

    logger.info(f"Calibration: methods={methods}")

    # Evaluate on holdout
    logger.info("Evaluating on holdout test set")
    y_test_pred_proba_raw = final_model.predict_proba(X_test_df)

    # CRITICAL: Remap to canonical order (train/serve parity)
    y_test_pred_proba_remapped = _remap_proba_via_classes(y_test_pred_proba_raw, final_model)

    # Apply calibration (matching inference behavior)
    from quant_core.prediction import apply_multiclass_calibration
    y_test_pred_proba_calibrated = apply_multiclass_calibration(
        y_test_pred_proba_remapped,
        final_model.calibrator_,
        final_model.calibration_methods_
    )

    # Compute logloss on CALIBRATED predictions (train/serve parity)
    holdout_logloss = log_loss(y_test, y_test_pred_proba_calibrated)
    logger.info(f"Holdout multi-logloss (calibrated): {holdout_logloss:.4f}")

    # Compute baseline logloss (always predict training class distribution)
    # Use np.bincount on y_train_full directly (robust for any 3-class task)
    priors = np.bincount(y_train_full, minlength=3) / len(y_train_full)
    baseline_pred = np.tile(priors, (len(y_test), 1))
    baseline_logloss = log_loss(y_test, baseline_pred, labels=[0, 1, 2])
    logloss_improvement = baseline_logloss - holdout_logloss

    logger.info(f"Baseline logloss (class priors): {baseline_logloss:.4f}")
    logger.info(f"Logloss improvement over baseline: {logloss_improvement:.4f}")

    return final_model, mean_cv_logloss, holdout_logloss, class_dist, optuna_best_params


def train_core(
    df: pd.DataFrame,
    horizons: list[int],
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
    min_train_bars_purged: int = 100,
) -> tuple[dict[str, Any], list[str], dict, dict]:
    """Train competing-risks models for range touch prediction.

    Trains one model per horizon predicting which barrier (UP_FIRST, DOWN_FIRST, TIMEOUT) is hit first.
    Uses time-series splits with purging/embargo based on bar_idx.

    Args:
        df: Multi-horizon labeled DataFrame (n_bars × n_configs rows)
        horizons: Prediction horizons in bars (e.g., [48, 96, 144, ...])
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
        min_train_bars_purged: Minimum training bars after purging

    Returns:
        Tuple of (models_dict, feature_cols, cv_metrics, holdout_metrics)
        - models_dict: {"lower_h48": model, "lower_h96": model, ..., "upper_h48": model, ...}
        - feature_cols: List of feature names (does NOT include "k")
        - cv_metrics: Cross-validation Brier scores per model
        - holdout_metrics: Holdout performance per model
    """
    from quant_tick.lib.schema import MLSchema

    logger.info(f"Starting multi-horizon model training for horizons: {horizons}")

    base_feature_cols = MLSchema.get_training_features(df.columns.tolist())

    frozen_feature_cols = base_feature_cols + [f"{c}_missing" for c in base_feature_cols]

    X, feature_cols = prepare_features(df, frozen_feature_cols)

    # Sanity check: ensure no label columns leaked into features
    label_leakage = [
        c for c in feature_cols
        if c.startswith("touched_")
        or c.startswith("first_hit_h")
        or c.startswith("hazard_")
        or c.startswith("event_")
    ]
    if label_leakage:
        raise ValueError(
            f"Label leakage detected! The following label columns are in feature_cols: {label_leakage}. "
            f"This will cause the model to cheat. Fix MLSchema.get_training_features()."
        )

    logger.info(f"Training with {len(feature_cols)} features (does NOT include k): {feature_cols[:10]}...")

    # Validate dataset structure (bars × configs)
    unique_bar_indices = df["bar_idx"].unique()
    n_bars = len(unique_bar_indices)
    n_configs = len(df[df["bar_idx"] == unique_bar_indices[0]])

    logger.info(
        f"Dataset structure: {n_bars} bars × {n_configs} configs = {len(df)} rows"
    )

    # Compute time-based splits for bars × configs dataset
    # Use bar-time splits (group by bar), not arbitrary row index
    max_horizon = max(horizons)
    test_bars = int(n_bars * holdout_pct)
    calib_bars = int(n_bars * calibration_pct)
    train_bars_initial = n_bars - test_bars - calib_bars

    # Purge bars whose events could overlap with calibration/test
    train_bars_purged = max(0, train_bars_initial - (max_horizon + 1))
    calib_bars_purged = max(0, calib_bars - (max_horizon + 1))

    if train_bars_purged < min_train_bars_purged:
        raise ValueError(
            f"Insufficient training data after purging: {train_bars_purged} bars "
            f"(min required: {min_train_bars_purged})"
        )

    # Get row indices for each split (bars × configs rows)
    train_idx = np.where(df["bar_idx"] < train_bars_purged)[0]

    first_calib_bar = train_bars_purged + max_horizon + 1
    calib_idx = np.where(
        (df["bar_idx"] >= first_calib_bar)
        & (df["bar_idx"] < first_calib_bar + calib_bars_purged)
    )[0]

    first_test_bar = first_calib_bar + calib_bars_purged + max_horizon + 1
    test_idx = np.where(df["bar_idx"] >= first_test_bar)[0]

    logger.info(
        f"Train bars: {train_bars_purged}, "
        f"Calib bars: {calib_bars_purged}, "
        f"Test bars: {n_bars - first_test_bar}"
    )

    X_train_full = X[train_idx]
    X_calib = X[calib_idx]
    X_test = X[test_idx]

    # Use bar_idx for purging (ensures time-ordering)
    bar_idx_train = df.iloc[train_idx]["bar_idx"].values

    models_dict = {}
    cv_metrics = {"cv_scores": {}, "optuna_best_params": {}}
    holdout_metrics = {"holdout_scores": {}, "class_distribution": {}}

    # Calculate n_jobs for LightGBM parallelism
    n_jobs = max((os.cpu_count() or 2) - 1, 1)

    # Multiclass parameters for competing-risks
    base_lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "min_child_samples": min_samples_leaf,
        "subsample": subsample,
        "subsample_freq": 1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": n_jobs,
    }

    # Train one multiclass model per horizon
    for H in horizons:
        target_col = f"first_hit_h{H}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Training first_hit_h{H} model (3-class)")
        logger.info(f"{'='*60}")

        model, cv_score, holdout_score, class_dist, optuna_params = _train_multiclass_model(
            model_name=f"first_hit_h{H}",
            df=df,
            X_train_full=X_train_full,
            X_calib=X_calib,
            X_test=X_test,
            train_idx=train_idx,
            calib_idx=calib_idx,
            test_idx=test_idx,
            bar_idx_train=bar_idx_train,
            max_horizon=H,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            base_lgbm_params=base_lgbm_params,
            optuna_n_trials=optuna_n_trials,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            label_col=target_col,
            feature_cols=feature_cols,
        )

        model_key = f"first_hit_h{H}"
        models_dict[model_key] = model
        cv_metrics["cv_scores"][model_key] = cv_score
        holdout_metrics["holdout_scores"][model_key] = holdout_score
        holdout_metrics["class_distribution"][model_key] = class_dist

        if optuna_params is not None:
            cv_metrics["optuna_best_params"][model_key] = optuna_params

    cv_metrics["avg_logloss"] = float(
        np.nanmean(list(cv_metrics["cv_scores"].values()))
    )
    holdout_metrics["avg_logloss"] = float(
        np.mean(list(holdout_metrics["holdout_scores"].values()))
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training complete")
    logger.info(f"Trained {len(models_dict)} models: {list(models_dict.keys())}")
    logger.info(f"Average CV log-loss: {cv_metrics['avg_logloss']:.4f}")
    logger.info(f"Average holdout log-loss: {holdout_metrics['avg_logloss']:.4f}")
    logger.info("=" * 60 + "\n")

    return models_dict, feature_cols, cv_metrics, holdout_metrics


def train_directional2_core(
    df_bars: pd.DataFrame,
    horizons: list[int],
    k: float = 0.25,
    vol_window: int = 48,
    n_splits: int = 5,
    embargo_bars: int = 96,
    holdout_pct: float = 0.2,
    calibration_pct: float = 0.1,
    n_estimators: int = 300,
    max_depth: int = 6,
    min_samples_leaf: int = 50,
    learning_rate: float = 0.05,
    subsample: float = 0.75,
    optuna_n_trials: int = 0,
) -> tuple[dict[str, Any], list[str], dict, dict]:
    """Train directional2 models for forward return prediction.

    Args:
        df_bars: DataFrame from load_feature_bars_df() (n_bars rows, no config grid)
        horizons: List of forward horizons (e.g., [48, 96])
        k: Volatility multiplier (for label metadata)
        vol_window: Vol window (for label metadata)
        n_splits: PurgedKFold splits
        embargo_bars: Embargo buffer
        holdout_pct: Final test set fraction
        calibration_pct: Calibration set fraction
        n_estimators: LightGBM n_estimators
        max_depth: LightGBM max_depth
        min_samples_leaf: LightGBM min_child_samples
        learning_rate: LightGBM learning_rate
        subsample: LightGBM subsample
        optuna_n_trials: Hyperparameter tuning trials (0=disable)

    Returns:
        Tuple of (models_dict, feature_cols, cv_metrics, holdout_metrics)
        - models_dict: {"directional_h48": model, "directional_h96": model, ...}
        - feature_cols: Frozen feature schema (includes *_missing)
        - cv_metrics: {"avg_logloss": X, "per_horizon": {48: X, 96: Y}}
        - holdout_metrics: Same structure as cv_metrics
    """
    from quant_core.prediction import prepare_features
    from quant_tick.lib.labels_directional import generate_directional_labels, label_stats
    from quant_tick.lib.schema import MLSchema

    logger.info(f"Training directional2 models for horizons: {horizons}")

    df = df_bars.copy()
    for H in horizons:
        labels = generate_directional_labels(df, H=H, vol_window=vol_window, k=k)
        df[f"label_h{H}"] = labels

        stats = label_stats(labels)
        logger.info(
            f"  H={H}: UP={stats['UP']:.1%}, DOWN={stats['DOWN']:.1%}, "
            f"FLAT={stats['FLAT']:.1%}, n={stats['n_samples']}"
        )

    label_cols = [f"label_h{H}" for H in horizons]
    df_labeled = df.dropna(subset=label_cols).reset_index(drop=True)

    # Convert labels back to int (they're float64 to support NaN)
    for label_col in label_cols:
        df_labeled[label_col] = df_labeled[label_col].astype(np.int8)

    n_bars = len(df_labeled)
    logger.info(
        f"Dataset: {n_bars} bars after dropping rows with missing labels "
        f"(dropped {len(df) - n_bars} rows)"
    )

    all_cols = df_labeled.columns.tolist()
    base_features = MLSchema.get_training_features(all_cols)
    base_features = [c for c in base_features if not c.startswith("label_h")]

    # Filter to numeric/bool columns only (LightGBM requirement)
    base_features = [
        c for c in base_features
        if pd.api.types.is_numeric_dtype(df_labeled[c]) or pd.api.types.is_bool_dtype(df_labeled[c])
    ]
    logger.info(f"Filtered to {len(base_features)} numeric/bool features")

    frozen_feature_cols = base_features + [f"{c}_missing" for c in base_features]
    X_array, feature_cols = prepare_features(df_labeled[base_features], frozen_feature_cols)

    logger.info(f"Frozen schema: {len(feature_cols)} columns (with *_missing)")

    n_train = int(n_bars * (1 - holdout_pct - calibration_pct))
    n_calib = int(n_bars * calibration_pct)

    train_idx = np.arange(n_train)
    calib_idx = np.arange(n_train, n_train + n_calib)
    test_idx = np.arange(n_train + n_calib, n_bars)

    logger.info(
        f"Split: train={len(train_idx)}, calib={len(calib_idx)}, test={len(test_idx)}"
    )

    X_train_full = X_array[train_idx]
    X_calib = X_array[calib_idx]
    X_test = X_array[test_idx]

    base_lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "force_col_wise": True,
    }

    models_dict = {}
    cv_scores = {}
    holdout_scores = {}

    for H in horizons:
        label_col = f"label_h{H}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Training directional_h{H} model")
        logger.info(f"{'='*60}")

        y_train = df_labeled[label_col].iloc[train_idx].values
        y_calib = df_labeled[label_col].iloc[calib_idx].values
        y_test = df_labeled[label_col].iloc[test_idx].values

        model, cv_score, holdout_score, class_dist, optuna_params = _train_multiclass_model(
            model_name=f"directional_h{H}",
            df=df_labeled,
            X_train_full=X_train_full,
            X_calib=X_calib,
            X_test=X_test,
            train_idx=train_idx,
            calib_idx=calib_idx,
            test_idx=test_idx,
            bar_idx_train=np.arange(len(train_idx)),
            max_horizon=H,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            base_lgbm_params=base_lgbm_params,
            optuna_n_trials=optuna_n_trials,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            label_col=label_col,
            feature_cols=feature_cols,
        )

        models_dict[f"directional_h{H}"] = model
        cv_scores[H] = cv_score
        holdout_scores[H] = holdout_score

        logger.info(f"  CV logloss: {cv_score:.4f}")
        logger.info(f"  Holdout logloss: {holdout_score:.4f}")
        logger.info(f"  Class dist: {class_dist}")

    cv_metrics = {
        "avg_logloss": np.mean(list(cv_scores.values())),
        "per_horizon": cv_scores,
    }

    holdout_metrics = {
        "avg_logloss": np.mean(list(holdout_scores.values())),
        "per_horizon": holdout_scores,
    }

    logger.info(f"\n{'='*60}")
    logger.info("Directional2 Training Complete")
    logger.info(f"CV Logloss (avg): {cv_metrics['avg_logloss']:.4f}")
    logger.info(f"Holdout Logloss (avg): {holdout_metrics['avg_logloss']:.4f}")
    logger.info(f"{'='*60}\n")

    return models_dict, feature_cols, cv_metrics, holdout_metrics


def train_models(config: MLConfig, output_path: str | None = None, **override_params) -> bool:
    """Train competing-risks models for MLConfig.

    Trains one multiclass model per horizon predicting {UP_FIRST, DOWN_FIRST, TIMEOUT}.

    Args:
        config: MLConfig with associated FeatureData (via config.candle)
        output_path: Optional GCS path (gs://bucket/path/model.joblib) to save bundle
        **override_params: Override any training parameter for this run
            (n_splits, embargo_bars, n_estimators, max_depth,
             min_samples_leaf, learning_rate, subsample, holdout_pct, calibration_pct,
             min_train_bars_purged)

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
    min_train_bars_purged = training_params["min_train_bars_purged"]

    # Get optuna trials from config
    optuna_n_trials = config.get_optuna_n_trials()

    # Load from FeatureData + generate labels on-the-fly
    from quant_tick.lib.feature_data import load_training_df

    try:
        df = load_training_df(config)
    except ValueError as e:
        logger.error(f"{config}: {e}")
        return False

    horizons = config.get_horizons()

    widths = config.get_widths()
    asymmetries = config.get_asymmetries()
    logger.info(
        f"{config}: loaded {len(df)} rows, "
        f"horizons={horizons}, "
        f"n_configs={len(widths) * len(asymmetries)}"
    )

    models_dict, feature_cols, cv_metrics, holdout_metrics = train_core(
        df,
        horizons=horizons,
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
        min_train_bars_purged=min_train_bars_purged,
    )

    if output_path:
        model_bundle = {
            "models": models_dict,
            "metadata": {
                "model_kind": "competing_risks",
                "model_type": "competing_risks",
                "feature_cols": feature_cols,
                "horizons": horizons,
                "widths": config.get_widths(),
                "asymmetries": config.get_asymmetries(),
                "trained_at": timezone.now().isoformat(),
                "model_version": "5.0",
            },
        }

        save_model_bundle_to_gcs(model_bundle, output_path)

        MLArtifact.objects.filter(
            ml_config=config,
            model_type="competing_risks_bundle",
        ).delete()

        MLArtifact.objects.create(
            ml_config=config,
            model_type="competing_risks_bundle",
            gcs_path=output_path,
            brier_score=holdout_metrics["avg_logloss"],
            feature_columns=feature_cols,
            json_data={
                "horizons": horizons,
                "cv_metrics": cv_metrics,
                "holdout_metrics": holdout_metrics,
            },
        )

        logger.info(f"Created bundled artifact with gcs_path={output_path}")
    else:
        logger.warning(
            "output_path is None - skipping model bundle save. "
            "Individual artifact saving is deprecated for competing-risks models."
        )

    config.json_data["model_kind"] = "competing_risks"
    config.json_data["model_version"] = "5.0"
    config.json_data["horizons"] = horizons
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
