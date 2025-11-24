import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quant_tick.constants import ExitReason, PositionStatus, PositionType
from quant_tick.lib.ml import (
    DEFAULT_ASYMMETRIES,
    DEFAULT_WIDTHS,
    LPConfig,
    check_position_change_allowed,
)
from quant_tick.models import MLArtifact, MLConfig, MLFeatureData, Position

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


def run_backtest(config: MLConfig) -> BacktestResult | None:
    """Run backtest simulation.

    Always persists Position records and clears existing backtest positions.

    Args:
        config: MLConfig to backtest

    Returns:
        BacktestResult if successful, None otherwise
    """
    # Read strategy params from config
    touch_tolerance = config.touch_tolerance
    min_hold_bars = config.min_hold_bars
    symbol = config.symbol

    # Load models
    artifact_lower = MLArtifact.objects.filter(
        ml_config=config, model_type="lower"
    ).order_by("-created_at").first()

    artifact_upper = MLArtifact.objects.filter(
        ml_config=config, model_type="upper"
    ).order_by("-created_at").first()

    if not artifact_lower or not artifact_upper:
        logger.error(f"{config}: missing model artifacts")
        return None

    model_lower = joblib.load(artifact_lower.artifact.open())
    model_upper = joblib.load(artifact_upper.artifact.open())
    feature_cols = artifact_lower.feature_columns

    logger.info(f"{config}: loaded models with {len(feature_cols)} features")

    # Load feature data
    feature_data = MLFeatureData.objects.filter(
        candle=config.candle
    ).order_by("-timestamp_to").first()

    if not feature_data or not feature_data.has_data_frame("file_data"):
        logger.error(f"{config}: no feature data available")
        return None

    df = feature_data.get_data_frame("file_data")

    # Clear existing backtest positions
    deleted, _ = Position.objects.filter(
        ml_config=config,
        position_type=PositionType.BACKTEST,
    ).delete()
    if deleted:
        logger.info(f"{config}: cleared {deleted} existing backtest positions")

    # Get widths/asymmetries from config
    widths = config.json_data.get("widths", DEFAULT_WIDTHS)
    asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)

    # Get base candle data (before config augmentation)
    n_configs = len(widths) * len(asymmetries)
    n_bars = len(df) // n_configs

    logger.info(f"{config}: {n_bars} bars, {n_configs} configs")

    # Run backtest
    result = _run_backtest_loop(
        config=config,
        symbol=symbol,
        df=df,
        model_lower=model_lower,
        model_upper=model_upper,
        feature_cols=feature_cols,
        widths=widths,
        asymmetries=asymmetries,
        touch_tolerance=touch_tolerance,
        min_hold_bars=min_hold_bars,
        n_bars=n_bars,
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
    model_lower: Any,
    model_upper: Any,
    feature_cols: list[str],
    widths: list[float],
    asymmetries: list[float],
    touch_tolerance: float,
    min_hold_bars: int,
    n_bars: int,
) -> BacktestResult:
    """Run backtest simulation loop."""
    # Track state
    current_config: LPConfig | None = None
    bars_since_change = 0

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

    for bar_idx in range(n_bars - 1):  # -1 to avoid lookahead on last bar
        # Get features for this bar across all configs (interleaved order)
        start = bar_idx * n_configs
        bar_rows = df.iloc[start : start + n_configs]

        if len(bar_rows) == 0:
            continue

        # Get timestamp and price for this bar
        ts = bar_rows["timestamp"].iloc[0] if "timestamp" in bar_rows.columns else None
        close_price = bar_rows["close"].iloc[0] if "close" in bar_rows.columns else None

        # Find optimal config for this bar
        best_config = _find_optimal_config(
            bar_rows=bar_rows,
            model_lower=model_lower,
            model_upper=model_upper,
            feature_cols=feature_cols,
            touch_tolerance=touch_tolerance,
        )

        # Check if we should change position
        if best_config is not None:
            should_change = current_config is None or (
                check_position_change_allowed(bars_since_change, min_hold_bars)
                and _config_significantly_different(current_config, best_config)
            )

            if should_change and current_config is not None:
                # Close existing position
                hold_lengths.append(bars_since_change)
                rebalance_count += 1

                if current_position_data is not None:
                    current_position_data["exit_timestamp"] = ts
                    current_position_data["exit_price"] = Decimal(str(close_price)) if close_price else None
                    current_position_data["exit_reason"] = ExitReason.REBALANCED
                    current_position_data["bars_held"] = bars_since_change
                    current_position_data["status"] = PositionStatus.CLOSED
                    positions_to_create.append(Position(**current_position_data))
                    current_position_data = None

                bars_since_change = 0

            if should_change:
                current_config = best_config

                # Create new position data
                current_position_data = {
                    "symbol": symbol,
                    "ml_config": config,
                    "position_type": PositionType.BACKTEST,
                    "lower_bound": best_config.lower_pct,
                    "upper_bound": best_config.upper_pct,
                    "borrow_ratio": best_config.borrow_ratio,
                    "entry_timestamp": ts,
                    "entry_price": Decimal(str(close_price)) if close_price else Decimal("0"),
                    "status": PositionStatus.OPEN,
                    "json_data": {
                        "p_touch_lower": best_config.p_touch_lower,
                        "p_touch_upper": best_config.p_touch_upper,
                        "width": best_config.width,
                        "asymmetry": best_config.asymmetry,
                    },
                }

        # Track metrics and check for touches
        if current_config is not None:
            bars_in_position += 1
            bars_since_change += 1

            # Check if we actually touched
            config_row = bar_rows[
                (bar_rows["width"] == current_config.width)
                & (bar_rows["asymmetry"] == current_config.asymmetry)
            ]
            if len(config_row) > 0:
                touched_lower = config_row["touched_lower"].iloc[0] == 1
                touched_upper = config_row["touched_upper"].iloc[0] == 1

                if touched_lower or touched_upper:
                    total_touches += 1

                    # Close position on touch
                    if current_position_data is not None:
                        current_position_data["exit_timestamp"] = ts
                        current_position_data["exit_price"] = Decimal(str(close_price)) if close_price else None
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
                else:
                    bars_in_range += 1

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
    model_lower: Any,
    model_upper: Any,
    feature_cols: list[str],
    touch_tolerance: float,
) -> LPConfig | None:
    """Find optimal config for a single bar."""
    # Prepare features for all configs - RF doesn't handle NaN, fill with 0
    X = bar_rows[feature_cols].fillna(0).values

    # Get predictions for all configs at once
    p_lower = model_lower.predict_proba(X)[:, 1]
    p_upper = model_upper.predict_proba(X)[:, 1]

    # Find best config (narrowest width that satisfies tolerance)
    best = None
    best_width = float("inf")

    for i, row in enumerate(bar_rows.itertuples()):
        if p_lower[i] < touch_tolerance and p_upper[i] < touch_tolerance:
            width = getattr(row, "width", None)
            asym = getattr(row, "asymmetry", None)

            if width is None or asym is None:
                continue

            if width < best_width:
                best_width = width
                lower_pct = -width * (0.5 - asym)
                upper_pct = width * (0.5 + asym)

                best = LPConfig(
                    lower_pct=lower_pct,
                    upper_pct=upper_pct,
                    borrow_ratio=0.5 + asym,
                    p_touch_lower=p_lower[i],
                    p_touch_upper=p_upper[i],
                    width=width,
                    asymmetry=asym,
                )

    return best


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
