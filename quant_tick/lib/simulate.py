"""Walk-forward simulation with rolling retrains."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from quant_tick.constants import ExitReason, PositionStatus, PositionType
from quant_tick.lib.ml import (
    DEFAULT_ASYMMETRIES,
    DEFAULT_WIDTHS,
    LPConfig,
    apply_calibration,
    check_position_change_allowed,
    compute_bound_features,
    enforce_monotonicity,
    prepare_features,
)
from quant_tick.lib.schema import MLSchema
from quant_tick.lib.train import train_model_core
from quant_tick.models import MLConfig, MLFeatureData, Position

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result summary."""

    total_bars: int
    bars_in_position: int
    bars_in_range: int
    total_touches: int
    touch_rate: float
    rebalances: int
    avg_hold_bars: float
    pct_in_range: float
    positions_created: int


@dataclass
class WalkForwardResult:
    """Walk-forward simulation result with observability."""

    aggregate_metrics: dict
    slice_results: list[dict]
    windows_attempted: int
    windows_skipped: int
    skip_reasons: dict[str, int]
    chronic_missing_features: dict[str, int]




def _run_backtest(
    config: MLConfig,
    lower_models: dict[int, tuple[Any, Any]],
    upper_models: dict[int, tuple[Any, Any]],
    feature_cols: list[str],
    df: pd.DataFrame,
    model_cutoff: Any | None = None,
    clear_positions: bool = True,
) -> BacktestResult | None:
    """Run backtest simulation on a single slice.

    Args:
        config: MLConfig to backtest
        lower_models: Dict of {horizon: (model, calibrator)} for lower bound
        upper_models: Dict of {horizon: (model, calibrator)} for upper bound
        feature_cols: Feature column names
        df: Feature DataFrame
        model_cutoff: Optional timestamp to tag positions with (for walk-forward)
        clear_positions: Whether to delete existing backtest positions (default: True)

    Returns:
        BacktestResult if successful, None otherwise
    """
    # Read strategy params from config
    touch_tolerance = config.touch_tolerance
    min_hold_bars = config.min_hold_bars
    symbol = config.symbol

    # Clear existing backtest positions if requested
    if clear_positions:
        deleted, _ = Position.objects.filter(
            ml_config=config,
            position_type=PositionType.BACKTEST,
        ).delete()
        if deleted:
            logger.info(f"{config}: cleared {deleted} existing backtest positions")

    # Get widths/asymmetries from config
    widths = config.json_data.get("widths", DEFAULT_WIDTHS)
    asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)
    decision_horizons = config.json_data.get("decision_horizons", [60, 120, 180])

    # Build bar-level price data for efficient lookup
    # Use GroupBy instead of arithmetic to handle reordered data
    n_configs = len(widths) * len(asymmetries)
    n_bars = len(df) // n_configs

    # Extract bar_idx -> close price mapping using GroupBy
    bar_prices = {}
    bar_timestamps = {}
    bar_groups = df.groupby("bar_idx")
    for bar_idx, bar_df in bar_groups:
        # Take first config (all have same close/timestamp at bar level)
        first_row = bar_df.iloc[0]
        if "close" in df.columns:
            bar_prices[bar_idx] = first_row["close"]
        if "timestamp" in df.columns:
            bar_timestamps[bar_idx] = first_row["timestamp"]

    logger.info(f"{config}: {n_bars} bars, {n_configs} configs")

    # Run backtest
    result = _run_backtest_loop(
        config=config,
        symbol=symbol,
        df=df,
        lower_models=lower_models,
        upper_models=upper_models,
        feature_cols=feature_cols,
        widths=widths,
        asymmetries=asymmetries,
        touch_tolerance=touch_tolerance,
        min_hold_bars=min_hold_bars,
        n_bars=n_bars,
        decision_horizons=decision_horizons,
        bar_prices=bar_prices,
        bar_timestamps=bar_timestamps,
        model_cutoff=model_cutoff,
    )

    # Print results
    logger.info(f"\n{'='*50}")
    logger.info(f"BACKTEST RESULTS: {config}")
    logger.info(f"{'='*50}")
    logger.info(f"Total bars: {result.total_bars}")
    logger.info(f"Bars in position: {result.bars_in_position}")
    logger.info(f"Bars in range: {result.bars_in_range}")
    logger.info(f"Total touches: {result.total_touches}")
    logger.info(f"Touch rate: {result.touch_rate:.2%}")
    logger.info(f"Rebalances: {result.rebalances}")
    logger.info(f"Avg hold bars: {result.avg_hold_bars:.1f}")
    logger.info(f"% in range: {result.pct_in_range:.2%}")
    logger.info(f"Positions saved: {result.positions_created}")
    logger.info(f"{'='*50}\n")

    return result


def _run_backtest_loop(
    config: MLConfig,
    symbol: Any,
    df: pd.DataFrame,
    lower_models: dict[int, tuple[Any, Any]],
    upper_models: dict[int, tuple[Any, Any]],
    feature_cols: list[str],
    widths: list[float],
    asymmetries: list[float],
    touch_tolerance: float,
    min_hold_bars: int,
    n_bars: int,
    decision_horizons: list[int],
    bar_prices: dict[int, float],
    bar_timestamps: dict[int, Any],
    model_cutoff: Any | None = None,
) -> BacktestResult:
    """Run backtest simulation loop using forward price checking."""
    # Track state
    current_config: LPConfig | None = None
    bars_since_change = 0
    position_entry_price: float | None = None
    position_lower_bound: float | None = None
    position_upper_bound: float | None = None

    # Metrics
    total_touches = 0
    bars_in_position = 0
    bars_in_range = 0
    rebalance_count = 0
    hold_lengths = []

    # Collect positions for bulk_create
    positions_to_create: list[Position] = []
    current_position_data: dict | None = None

    n_configs = len(widths) * len(asymmetries)

    for bar_idx in range(n_bars):
        # Get timestamp and price for this bar
        ts = bar_timestamps.get(bar_idx)
        close_price = bar_prices.get(bar_idx)

        if close_price is None:
            continue

        # If position is open, check for exit conditions
        if current_config is not None and position_entry_price is not None:
            bars_in_position += 1
            bars_since_change += 1

            # Check for max horizon timeout first
            if bars_since_change >= config.horizon_bars:
                # Close position - reached decision horizon
                if current_position_data is not None:
                    current_position_data["exit_timestamp"] = ts
                    current_position_data["exit_price"] = Decimal(str(close_price))
                    current_position_data["exit_reason"] = ExitReason.MAX_DURATION
                    current_position_data["bars_held"] = bars_since_change
                    current_position_data["status"] = PositionStatus.CLOSED
                    positions_to_create.append(Position(**current_position_data))
                    current_position_data = None

                hold_lengths.append(bars_since_change)
                bars_since_change = 0
                current_config = None
                position_entry_price = None
                position_lower_bound = None
                position_upper_bound = None

            else:
                # Check if current price touches bounds (using forward prices, not labels)
                touched_lower = close_price <= position_lower_bound
                touched_upper = close_price >= position_upper_bound

                if touched_lower or touched_upper:
                    total_touches += 1

                    # Close position on touch
                    if current_position_data is not None:
                        current_position_data["exit_timestamp"] = ts
                        current_position_data["exit_price"] = Decimal(str(close_price))
                        current_position_data["exit_reason"] = (
                            ExitReason.TOUCHED_LOWER if touched_lower else ExitReason.TOUCHED_UPPER
                        )
                        current_position_data["bars_held"] = bars_since_change
                        current_position_data["status"] = PositionStatus.CLOSED
                        positions_to_create.append(Position(**current_position_data))
                        current_position_data = None

                    hold_lengths.append(bars_since_change)
                    bars_since_change = 0
                    current_config = None
                    position_entry_price = None
                    position_lower_bound = None
                    position_upper_bound = None
                else:
                    bars_in_range += 1

        # If no position, or position just closed, check if we should open/rebalance
        if current_config is None or (
            current_config is not None
            and check_position_change_allowed(bars_since_change, min_hold_bars)
        ):
            # Get features for this bar across all configs
            bar_rows = df[df["bar_idx"] == bar_idx]

            if len(bar_rows) == 0:
                continue

            # Validate complete config set
            if len(bar_rows) != n_configs:
                logger.warning(
                    f"Bar {bar_idx} has {len(bar_rows)} configs, expected {n_configs}, skipping"
                )
                continue

            # Find optimal config for this bar
            best_config = _find_optimal_config(
                bar_rows=bar_rows,
                lower_models=lower_models,
                upper_models=upper_models,
                feature_cols=feature_cols,
                touch_tolerance=touch_tolerance,
                decision_horizons=decision_horizons,
            )

            if best_config is None:
                continue

            # Check if we should change position
            should_change = current_config is None or _config_significantly_different(
                current_config, best_config
            )

            if should_change:
                # Close existing position if any (rebalance)
                if current_config is not None and current_position_data is not None:
                    current_position_data["exit_timestamp"] = ts
                    current_position_data["exit_price"] = Decimal(str(close_price))
                    current_position_data["exit_reason"] = ExitReason.REBALANCED
                    current_position_data["bars_held"] = bars_since_change
                    current_position_data["status"] = PositionStatus.CLOSED
                    positions_to_create.append(Position(**current_position_data))
                    current_position_data = None

                    hold_lengths.append(bars_since_change)
                    rebalance_count += 1
                    bars_since_change = 0

                # Open new position
                current_config = best_config
                position_entry_price = close_price
                position_lower_bound = position_entry_price * (1 + best_config.lower_pct)
                position_upper_bound = position_entry_price * (1 + best_config.upper_pct)

                # Create new position data
                json_data_dict = {
                    "p_touch_lower": best_config.p_touch_lower,
                    "p_touch_upper": best_config.p_touch_upper,
                    "width": best_config.width,
                    "asymmetry": best_config.asymmetry,
                }
                if model_cutoff is not None:
                    json_data_dict["model_cutoff"] = str(model_cutoff)

                current_position_data = {
                    "symbol": symbol,
                    "ml_config": config,
                    "position_type": PositionType.BACKTEST,
                    "lower_bound": best_config.lower_pct,
                    "upper_bound": best_config.upper_pct,
                    "borrow_ratio": best_config.borrow_ratio,
                    "entry_timestamp": ts,
                    "entry_price": Decimal(str(close_price)),
                    "status": PositionStatus.OPEN,
                    "json_data": json_data_dict,
                }

    # Close any remaining open position
    if current_position_data is not None:
        current_position_data["exit_reason"] = ExitReason.MAX_DURATION
        current_position_data["bars_held"] = bars_since_change
        current_position_data["status"] = PositionStatus.CLOSED
        positions_to_create.append(Position(**current_position_data))

    # Final hold length
    if current_config is not None and bars_since_change > 0:
        hold_lengths.append(bars_since_change)

    # Bulk create all positions
    Position.objects.bulk_create(positions_to_create)
    positions_created = len(positions_to_create)

    touch_rate = total_touches / bars_in_position if bars_in_position > 0 else 0
    avg_hold = np.mean(hold_lengths) if hold_lengths else 0
    pct_in_range = bars_in_range / bars_in_position if bars_in_position > 0 else 0

    return BacktestResult(
        total_bars=n_bars,
        bars_in_position=bars_in_position,
        bars_in_range=bars_in_range,
        total_touches=total_touches,
        touch_rate=touch_rate,
        rebalances=rebalance_count,
        avg_hold_bars=avg_hold,
        pct_in_range=pct_in_range,
        positions_created=positions_created,
    )


def _find_optimal_config(
    bar_rows: pd.DataFrame,
    lower_models: dict[int, tuple[Any, Any, str]],
    upper_models: dict[int, tuple[Any, Any, str]],
    feature_cols: list[str],
    touch_tolerance: float,
    decision_horizons: list[int],
) -> LPConfig | None:
    """Adapter to use canonical find_optimal_config from ml.py with simulate.py models.

    This replaces duplicate logic by wrapping the canonical implementation with
    predict functions that use the loaded models.
    """
    from quant_tick.lib.ml import find_optimal_config

    # Extract widths and asymmetries from bar_rows
    widths = sorted(bar_rows["width"].unique()) if "width" in bar_rows.columns else None
    asymmetries = sorted(bar_rows["asymmetry"].unique()) if "asymmetry" in bar_rows.columns else None

    # Get first row as features (all rows have same base features, just different width/asymmetry)
    features = bar_rows.iloc[[0]].copy()

    # Create prediction functions that use the loaded models
    def predict_lower(feat: pd.DataFrame, lower_pct: float, upper_pct: float) -> float:
        """Predict max risk across horizons for lower bound."""
        feat_with_bounds = compute_bound_features(feat, lower_pct, upper_pct)
        X_inference, _ = prepare_features(feat_with_bounds, feature_cols)

        horizon_probs = {}
        for h, (model, calibrator, calibration_method) in lower_models.items():
            proba = model.predict_proba(X_inference)[0, 1]
            proba = apply_calibration(proba, calibrator, calibration_method)
            horizon_probs[h] = proba

        horizon_probs = enforce_monotonicity(horizon_probs)
        return float(max(horizon_probs.values()))

    def predict_upper(feat: pd.DataFrame, lower_pct: float, upper_pct: float) -> float:
        """Predict max risk across horizons for upper bound."""
        feat_with_bounds = compute_bound_features(feat, lower_pct, upper_pct)
        X_inference, _ = prepare_features(feat_with_bounds, feature_cols)

        horizon_probs = {}
        for h, (model, calibrator, calibration_method) in upper_models.items():
            proba = model.predict_proba(X_inference)[0, 1]
            proba = apply_calibration(proba, calibrator, calibration_method)
            horizon_probs[h] = proba

        horizon_probs = enforce_monotonicity(horizon_probs)
        return float(max(horizon_probs.values()))

    # Use canonical implementation from ml.py
    return find_optimal_config(
        predict_lower_fn=predict_lower,
        predict_upper_fn=predict_upper,
        features=features,
        touch_tolerance=touch_tolerance,
        widths=widths,
        asymmetries=asymmetries,
    )


def _config_significantly_different(
    current: LPConfig,
    new: LPConfig,
    width_threshold: float = 0.01,
    asym_threshold: float = 0.1,
) -> bool:
    """Check if new config is significantly different from current."""
    width_diff = abs(new.width - current.width)
    asym_diff = abs(new.asymmetry - current.asymmetry)

    return width_diff > width_threshold or asym_diff > asym_threshold


def ml_simulate(
    config: MLConfig,
    retrain_cadence_days: int = 7,
    train_window_days: int = 84,
    holdout_days: int | None = None,
    **backtest_kwargs,
) -> dict:
    """Run walk-forward simulation with rolling retrains.

    Args:
        config: MLConfig to simulate
        retrain_cadence_days: Days between retrains
        train_window_days: Days of data to train on
        holdout_days: Optional final holdout period (not used for any training)
        **backtest_kwargs: Additional args passed to backtest

    Returns:
        Dict with aggregated metrics and per-slice results
    """
    # Load all feature data
    feature_data = MLFeatureData.objects.filter(
        candle=config.candle
    ).order_by("-timestamp_to").first()

    if not feature_data or not feature_data.has_data_frame("file_data"):
        logger.error(f"{config}: no feature data available")
        return {}

    df = feature_data.get_data_frame("file_data")
    logger.info(f"{config}: loaded {len(df)} rows for walk-forward simulation")

    # Extract config parameters
    decision_horizons = config.json_data.get("decision_horizons", [60, 120, 180])
    widths = config.json_data.get("widths", DEFAULT_WIDTHS)
    asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)

    # Determine time range
    if "timestamp" in df.columns:
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
    else:
        logger.error("DataFrame missing timestamp column")
        return {}

    # Reserve holdout if specified
    if holdout_days:
        holdout_start = max_ts - timedelta(days=holdout_days)
        df = df[df["timestamp"] < holdout_start]
        max_ts = df["timestamp"].max()
        logger.info(f"{config}: reserved {holdout_days} days as final holdout")

    # Generate cutoff dates
    train_window_delta = timedelta(days=train_window_days)
    cadence_delta = timedelta(days=retrain_cadence_days)

    # Auto-detect start: validation begins after first training window
    cutoffs = []
    cutoff = min_ts + train_window_delta
    while cutoff + cadence_delta <= max_ts:
        cutoffs.append(cutoff)
        cutoff += cadence_delta

    logger.info(f"{config}: {len(cutoffs)} training windows")

    # Clear all existing backtest positions at the start
    deleted, _ = Position.objects.filter(
        ml_config=config,
        position_type=PositionType.BACKTEST,
    ).delete()
    if deleted:
        logger.info(f"{config}: cleared {deleted} existing backtest positions")

    # Track skipped windows for observability
    slice_results = []
    skip_reasons = {}
    missing_feature_counts = {}
    windows_attempted = len(cutoffs)
    windows_skipped = 0

    # Track base rates across windows for drift detection
    window_base_rates = {}

    for i, cutoff in enumerate(cutoffs):
        logger.info(f"{config}: window {i+1}/{len(cutoffs)} - cutoff={cutoff}")

        # Training slice: window before cutoff
        train_start = cutoff - train_window_delta
        train_df = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= cutoff)]

        if len(train_df) == 0:
            logger.warning(f"{config}: no training data for cutoff {cutoff}")
            windows_skipped += 1
            skip_reasons["no_training_data"] = skip_reasons.get("no_training_data", 0) + 1
            continue

        # Validate complete bar/config structure after timestamp filtering (if columns exist)
        if "bar_idx" in train_df.columns and "config_id" in train_df.columns:
            is_valid, error = MLSchema.validate_bar_config_structure(train_df, widths, asymmetries)
            if not is_valid:
                logger.error(f"{config}: Training window invalid - {error}, skipping")
                windows_skipped += 1
                skip_reasons["bar_config_structure_invalid"] = skip_reasons.get("bar_config_structure_invalid", 0) + 1
                continue

        # Train models on window
        try:
            models_dict, feature_cols, cv_metrics, holdout_metrics = train_model_core(
                df=train_df,
                decision_horizons=decision_horizons,
                n_splits=3,  # Fewer splits for speed
                embargo_bars=96,
                holdout_pct=0.15,  # Smaller holdout for walk-forward
                optuna_n_trials=0,  # Never run Optuna in walk-forward (too slow)
            )
        except Exception as e:
            logger.error(f"{config}: training failed for cutoff {cutoff}: {e}")
            windows_skipped += 1
            skip_reasons["training_failed"] = skip_reasons.get("training_failed", 0) + 1
            continue

        # Extract base rates for rare event monitoring
        base_rates = holdout_metrics.get("base_rates", {})

        # Track base rates across windows
        if i == 0:
            # First window sets baseline
            window_base_rates["baseline"] = base_rates.copy()
        else:
            # Compare current window to baseline
            baseline_rates = window_base_rates.get("baseline", {})
            for model_key, current_rate in base_rates.items():
                baseline_rate = baseline_rates.get(model_key)
                if baseline_rate is not None and baseline_rate > 0:
                    drift_ratio = abs(current_rate - baseline_rate) / baseline_rate
                    # Warn if base rate changed by >30%
                    if drift_ratio > 0.3:
                        logger.warning(
                            f"{config}: {model_key} base rate drift detected - "
                            f"baseline={baseline_rate:.4f}, current={current_rate:.4f} "
                            f"({drift_ratio:.1%} change)"
                        )

        # Extract per-horizon models with calibrators
        lower_models = {}
        upper_models = {}
        for h in decision_horizons:
            lower_key = f"lower_h{h}"
            upper_key = f"upper_h{h}"

            if lower_key not in models_dict or upper_key not in models_dict:
                logger.error(f"{config}: missing models for horizon {h}")
                continue

            lower_model = models_dict[lower_key]
            upper_model = models_dict[upper_key]

            # Get calibrators and calibration methods from model attributes
            lower_cal = getattr(lower_model, "calibrator_", None)
            upper_cal = getattr(upper_model, "calibrator_", None)
            lower_cal_method = getattr(lower_model, "calibration_method_", "none")
            upper_cal_method = getattr(upper_model, "calibration_method_", "none")

            lower_models[h] = (lower_model, lower_cal, lower_cal_method)
            upper_models[h] = (upper_model, upper_cal, upper_cal_method)

        # Scoring slice: next cadence period
        score_start = cutoff
        score_end = cutoff + cadence_delta
        score_df = df[(df["timestamp"] > score_start) & (df["timestamp"] <= score_end)]

        if len(score_df) == 0:
            logger.warning(f"{config}: no scoring data for cutoff {cutoff}")
            windows_skipped += 1
            skip_reasons["no_scoring_data"] = skip_reasons.get("no_scoring_data", 0) + 1
            continue

        # Validate scoring window grid structure
        if "bar_idx" in score_df.columns and "config_id" in score_df.columns:
            is_valid, error = MLSchema.validate_bar_config_structure(score_df, widths, asymmetries)
            if not is_valid:
                logger.warning(f"Window {cutoff}: scoring data invalid - {error}")
                windows_skipped += 1
                skip_reasons["scoring_bar_config_invalid"] = skip_reasons.get("scoring_bar_config_invalid", 0) + 1
                continue

        # Validate features match using centralized schema
        # Use centralized schema to get data features (excludes config cols added during prediction)
        data_feature_cols = MLSchema.get_data_features(feature_cols, decision_horizons)
        available_cols = set(score_df.columns)
        missing_cols = [c for c in data_feature_cols if c not in available_cols]
        if missing_cols:
            logger.error(
                f"{config}: Cannot run backtest - {len(missing_cols)} required feature(s) missing: "
                f"{', '.join(missing_cols[:10])}{'...' if len(missing_cols) > 10 else ''}. "
                f"Skipping this window to prevent schema mismatch."
            )
            windows_skipped += 1
            skip_reasons["missing_features"] = skip_reasons.get("missing_features", 0) + 1

            # Track which features are chronic offenders
            for col in missing_cols:
                missing_feature_counts[col] = missing_feature_counts.get(col, 0) + 1
            continue

        # Run backtest on scoring slice with trained models (don't clear positions from previous slices)
        backtest_result = _run_backtest(
            config=config,
            lower_models=lower_models,
            upper_models=upper_models,
            feature_cols=feature_cols,
            df=score_df,
            model_cutoff=cutoff,
            clear_positions=False,
        )

        if backtest_result:
            slice_results.append({
                "cutoff": cutoff,
                "train_size": len(train_df),
                "score_size": len(score_df),
                "cv_brier_lower": cv_metrics.get("avg_brier_lower", 0.0),
                "cv_brier_upper": cv_metrics.get("avg_brier_upper", 0.0),
                "holdout_brier_lower": holdout_metrics.get("avg_brier_lower", 0.0),
                "holdout_brier_upper": holdout_metrics.get("avg_brier_upper", 0.0),
                "backtest_touch_rate": backtest_result.touch_rate,
                "backtest_pct_in_range": backtest_result.pct_in_range,
                "backtest_rebalances": backtest_result.rebalances,
                "backtest_positions": backtest_result.positions_created,
            })

            logger.info(
                f"{config}: window {i+1} - CV brier: {cv_metrics.get('avg_brier_lower', 0):.4f}/{cv_metrics.get('avg_brier_upper', 0):.4f}, "
                f"holdout brier: {holdout_metrics.get('avg_brier_lower', 0):.4f}/{holdout_metrics.get('avg_brier_upper', 0):.4f}, "
                f"touch_rate: {backtest_result.touch_rate:.2%}"
            )

    # Aggregate metrics
    if slice_results:
        agg_metrics = {
            "n_windows": len(slice_results),
            "avg_cv_brier_lower": sum(r["cv_brier_lower"] for r in slice_results) / len(slice_results),
            "avg_cv_brier_upper": sum(r["cv_brier_upper"] for r in slice_results) / len(slice_results),
            "avg_holdout_brier_lower": sum(r["holdout_brier_lower"] for r in slice_results) / len(slice_results),
            "avg_holdout_brier_upper": sum(r["holdout_brier_upper"] for r in slice_results) / len(slice_results),
            "avg_touch_rate": sum(r["backtest_touch_rate"] for r in slice_results) / len(slice_results),
            "avg_pct_in_range": sum(r["backtest_pct_in_range"] for r in slice_results) / len(slice_results),
            "total_rebalances": sum(r["backtest_rebalances"] for r in slice_results),
            "total_positions": sum(r["backtest_positions"] for r in slice_results),
        }
    else:
        agg_metrics = {}

    # Log walk-forward data loss summary
    if windows_skipped > 0:
        logger.warning(
            f"\n{'='*60}\n"
            f"WALK-FORWARD DATA LOSS SUMMARY\n"
            f"{'='*60}\n"
            f"Windows attempted: {windows_attempted}\n"
            f"Windows skipped: {windows_skipped} ({windows_skipped/windows_attempted:.1%})\n"
            f"\nSkip reasons:"
        )
        for reason, count in skip_reasons.items():
            logger.warning(f"  - {reason}: {count} windows")

        if missing_feature_counts:
            chronic_threshold = windows_attempted * 0.2
            chronic_features = {
                feat: count
                for feat, count in missing_feature_counts.items()
                if count >= chronic_threshold
            }

            if chronic_features:
                logger.warning("\nChronic missing features (>20% of windows):")
                for feat, count in sorted(chronic_features.items(), key=lambda x: -x[1]):
                    pct = count / windows_attempted
                    logger.warning(f"  - {feat}: missing in {count}/{windows_attempted} windows ({pct:.1%})")

    return WalkForwardResult(
        aggregate_metrics=agg_metrics,
        slice_results=slice_results,
        windows_attempted=windows_attempted,
        windows_skipped=windows_skipped,
        skip_reasons=skip_reasons,
        chronic_missing_features=missing_feature_counts,
    )
