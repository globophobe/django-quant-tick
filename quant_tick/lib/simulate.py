import logging
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from quant_core.constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from quant_core.prediction import LPConfig
from tqdm import tqdm

from quant_tick.constants import ExitReason, PositionStatus, PositionType
from quant_tick.lib.feature_data import load_training_df
from quant_tick.lib.ml import check_position_change_allowed
from quant_tick.lib.schema import MLSchema
from quant_tick.lib.train import train_core
from quant_tick.models import MLConfig, Position

logger = logging.getLogger(__name__)


def _record_perps_exit_metrics(
    position_direction: str,
    exit_price: float,
    position_entry_price: float,
    position_peak: float,
    wins: list,
    losses: list,
    mfes: list,
) -> None:
    """Record PnL and MFE metrics for perps position exit."""
    # PnL
    pnl_pct = (
        (exit_price - position_entry_price) / position_entry_price
        if position_direction == "long"
        else (position_entry_price - exit_price) / position_entry_price
    )
    if pnl_pct > 0:
        wins.append(pnl_pct)
    else:
        losses.append(abs(pnl_pct))

    # MFE
    mfe = (
        (position_peak - position_entry_price) / position_entry_price
        if position_direction == "long"
        else (position_entry_price - position_peak) / position_entry_price
    )
    mfes.append(mfe)


@dataclass
class BacktestResult:
    """Backtest result."""

    total_bars: int
    bars_in_position: int
    bars_in_range: int
    total_touches: int
    touch_rate: float
    rebalances: int
    avg_hold_bars: float
    pct_in_range: float
    positions_created: int
    bars_with_no_valid_config: int = 0
    perps_metrics: dict | None = None
    lp_metrics: dict | None = None
    directional_metrics: dict | None = None


@dataclass
class WalkForwardResult:
    """Walk-forward result."""

    aggregate_metrics: dict
    slice_results: list[dict]
    windows_attempted: int
    windows_skipped: int
    skip_reasons: dict[str, int]
    chronic_missing_features: dict[str, int]
    bars_with_no_valid_config: int = 0


def _run_backtest(
    config: MLConfig,
    models_dict: dict[str, Any],
    feature_cols: list[str],
    df: pd.DataFrame,
    model_cutoff: Any | None = None,
    clear_positions: bool = True,
    policy_mode: str = "lp",
) -> BacktestResult | None:
    """Run backtest on a single slice.

    Args:
        config: MLConfig to backtest
        models_dict: Multi-horizon model bundle
        feature_cols: Feature column names (excludes k)
        df: Feature DataFrame
        model_cutoff: Optional timestamp, for walk-forward, to tag positions with
        clear_positions: Whether to delete existing backtest positions

    Returns:
        BacktestResult if successful, None otherwise
    """
    # Read strategy params from config
    touch_tolerance = config.touch_tolerance
    min_hold_bars = config.min_hold_bars
    symbol = config.symbol

    # Clear existing backtest positions
    if clear_positions:
        deleted, _ = Position.objects.filter(
            ml_config=config,
            position_type=PositionType.BACKTEST,
        ).delete()
        if deleted:
            logger.info(f"{config}: cleared {deleted} existing backtest positions")

    # Get config - both modes use same training grid
    widths = config.json_data.get("widths", DEFAULT_WIDTHS)
    asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)

    if policy_mode == "perps":
        perps_params = config.get_perps_params()
        conf_threshold = perps_params["conf_threshold"]
        move_threshold = perps_params["move_threshold"]
        trail_pct = perps_params["trail_pct"]
        perps_widths = perps_params["widths"]
        exposures = None
        skew_threshold = None
        max_abs_exposure = None
    elif policy_mode == "directional":
        conf_threshold = None
        move_threshold = None
        trail_pct = None
        perps_widths = None
        exposures = config.get_exposures()
        skew_threshold = config.json_data.get("skew_threshold", 0.05)
        max_abs_exposure = config.json_data.get("max_abs_exposure", 1.0)
    else:
        conf_threshold = None
        move_threshold = None
        trail_pct = None
        perps_widths = None
        exposures = None
        skew_threshold = None
        max_abs_exposure = None

    # Get horizons from config (no backwards compat)
    horizons = config.get_horizons()

    # Build bar-level price data for efficient lookup
    n_configs = len(widths) * len(asymmetries)
    n_bars = len(df) // n_configs

    # Extract bar_idx -> close price and OHLC mapping
    bar_prices = {}
    bar_ohlc = {}
    bar_timestamps = {}
    bar_groups = df.groupby("bar_idx")
    for bar_idx, bar_df in bar_groups:
        # Take first config
        first_row = bar_df.iloc[0]
        if "close" in df.columns:
            bar_prices[bar_idx] = first_row["close"]
            # Build OHLC once for efficient lookup
            bar_ohlc[bar_idx] = (
                first_row.get("open", first_row["close"]),
                first_row.get("high", first_row["close"]),
                first_row.get("low", first_row["close"]),
                first_row["close"]
            )
        if "timestamp" in df.columns:
            bar_timestamps[bar_idx] = first_row["timestamp"]

    logger.info(f"{config}: {n_bars} bars, {n_configs} configs")

    # Run backtest
    result = _run_backtest_loop(
        config=config,
        symbol=symbol,
        df=df,
        models_dict=models_dict,
        feature_cols=feature_cols,
        widths=widths,
        asymmetries=asymmetries,
        touch_tolerance=touch_tolerance,
        min_hold_bars=min_hold_bars,
        n_bars=n_bars,
        horizons=horizons,
        bar_prices=bar_prices,
        bar_ohlc=bar_ohlc,
        bar_timestamps=bar_timestamps,
        model_cutoff=model_cutoff,
        policy_mode=policy_mode,
        conf_threshold=conf_threshold,
        move_threshold=move_threshold,
        trail_pct=trail_pct,
        perps_widths=perps_widths,
        exposures=exposures,
        skew_threshold=skew_threshold,
        max_abs_exposure=max_abs_exposure,
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
    logger.info(f"Bars with no valid config: {result.bars_with_no_valid_config}")
    logger.info(f"{'='*50}\n")

    return result


def _run_backtest_loop(
    config: MLConfig,
    symbol: Any,
    df: pd.DataFrame,
    models_dict: dict[str, Any],
    feature_cols: list[str],
    widths: list[float],
    asymmetries: list[float],
    touch_tolerance: float,
    min_hold_bars: int,
    n_bars: int,
    horizons: list[int],
    bar_prices: dict[int, float],
    bar_ohlc: dict[int, tuple[float, float, float, float]],
    bar_timestamps: dict[int, Any],
    model_cutoff: Any | None = None,
    policy_mode: str = "lp",
    conf_threshold: float | None = None,
    move_threshold: float | None = None,
    trail_pct: float | None = None,
    perps_widths: list[float] | None = None,
    exposures: list[float] | None = None,
    skew_threshold: float | None = None,
    max_abs_exposure: float | None = None,
) -> BacktestResult:
    """Run backtest loop."""
    # Track state
    current_config: LPConfig | None = None
    bars_since_change = 0
    position_entry_price: float | None = None
    position_lower_bound: float | None = None
    position_upper_bound: float | None = None

    # Perps state
    position_direction: str | None = None
    position_peak: float | None = None
    position_stop: float | None = None

    # Metrics
    total_touches = 0
    bars_in_position = 0
    bars_in_range = 0
    rebalance_count = 0
    hold_lengths = []
    no_config_count = 0

    # Perps metrics
    n_trailing_stops = 0
    wins = []
    losses = []
    mfes = []

    # LP aggregates
    lp_n_trades = 0
    lp_sum_ret = 0.0
    lp_log_compound = 0.0
    lp_by_exit = {}

    # Directional aggregates
    dir_n_trades = 0
    dir_sum_ret = 0.0
    dir_log_compound = 0.0
    dir_by_exit = {}

    # Collect positions for bulk_create
    positions_to_create: list[Position] = []
    current_position_data: dict | None = None

    from django.utils import timezone as tz

    def ensure_aware_timestamp(timestamp: Any) -> Any:
        """Make timestamp timezone-aware if naive to avoid Django warnings."""
        return timestamp if tz.is_aware(timestamp) else tz.make_aware(timestamp)

    n_configs = len(widths) * len(asymmetries)

    for bar_idx in tqdm(range(n_bars), desc="Backtest bars", leave=False):
        # Get timestamp and price for this bar
        ts = bar_timestamps.get(bar_idx)
        close_price = bar_prices.get(bar_idx)

        if close_price is None:
            continue

        # Extract OHLC from pre-built dict (efficient)
        bar_open, bar_high, bar_low, bar_close = bar_ohlc.get(
            bar_idx, (close_price, close_price, close_price, close_price)
        )
        close_price = bar_close

        # If position is open, check for exit conditions
        if current_config is not None and position_entry_price is not None:
            bars_in_position += 1
            bars_since_change += 1

            # Check for max horizon timeout first
            if bars_since_change >= config.horizon_bars:
                # Close position - reached decision horizon
                if current_position_data is not None:
                    current_position_data["exit_timestamp"] = ensure_aware_timestamp(ts)
                    current_position_data["exit_price"] = Decimal(str(bar_close))
                    current_position_data["exit_reason"] = ExitReason.MAX_DURATION
                    current_position_data["bars_held"] = bars_since_change
                    current_position_data["status"] = PositionStatus.CLOSED

                    # Record perps metrics (using bar_close as mark/exit assumption)
                    if policy_mode == "perps" and position_direction is not None:
                        _record_perps_exit_metrics(
                            position_direction, bar_close, position_entry_price,
                            position_peak, wins, losses, mfes
                        )

                    # Track LP return for this position (LP mode only)
                    if policy_mode == "lp":
                        entry_price = float(current_position_data["entry_price"])
                        exit_price = float(current_position_data["exit_price"])
                        ret = (exit_price - entry_price) / entry_price

                        exit_reason = str(current_position_data["exit_reason"])

                        lp_n_trades += 1
                        lp_sum_ret += ret

                        ret_factor = max(1 + ret, 1e-10)
                        lp_log_compound += np.log(ret_factor)

                        if exit_reason not in lp_by_exit:
                            lp_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
                        lp_by_exit[exit_reason]["count"] += 1
                        lp_by_exit[exit_reason]["sum_ret"] += ret

                    # Track directional return for this position (directional mode only)
                    if policy_mode == "directional":
                        entry_price = float(current_position_data["entry_price"])
                        exit_price = float(current_position_data["exit_price"])

                        # Retrieve exposure from json_data
                        json_data = current_position_data.get("json_data", {})
                        exposure = json_data.get("exposure", 0.0)

                        # Compute return using exposure formula: ret = e * (P1/P0 - 1)
                        price_return = exit_price / entry_price - 1
                        ret = exposure * price_return

                        exit_reason = str(current_position_data["exit_reason"])

                        dir_n_trades += 1
                        dir_sum_ret += ret

                        ret_factor = max(1 + ret, 1e-10)
                        dir_log_compound += np.log(ret_factor)

                        if exit_reason not in dir_by_exit:
                            dir_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
                        dir_by_exit[exit_reason]["count"] += 1
                        dir_by_exit[exit_reason]["sum_ret"] += ret

                    positions_to_create.append(Position(**current_position_data))
                    current_position_data = None

                # CRITICAL: Reset ALL state including perps state
                hold_lengths.append(bars_since_change)
                bars_since_change = 0
                current_config = None
                position_entry_price = None
                position_lower_bound = None
                position_upper_bound = None
                if policy_mode == "perps":
                    position_direction = None
                    position_peak = None
                    position_stop = None

            else:
                # Trailing stop logic (perps mode only)
                if policy_mode == "perps" and position_direction is not None:
                    # 1. Check stop hit using PREVIOUS stop level (before updating)
                    if position_stop is not None:
                        stop_hit = False
                        if position_direction == "long" and bar_low <= position_stop:
                            stop_hit = True
                        elif position_direction == "short" and bar_high >= position_stop:
                            stop_hit = True

                        if stop_hit:
                            n_trailing_stops += 1

                            # Conservative fill: worse of stop vs close
                            if position_direction == "long":
                                exit_price = min(position_stop, bar_close)
                            else:
                                exit_price = max(position_stop, bar_close)

                            # Close position
                            if current_position_data is not None:
                                current_position_data["exit_timestamp"] = ensure_aware_timestamp(ts)
                                current_position_data["exit_price"] = Decimal(str(exit_price))
                                current_position_data["exit_reason"] = ExitReason.TRAILING_STOP
                                current_position_data["bars_held"] = bars_since_change
                                current_position_data["status"] = PositionStatus.CLOSED
                                current_position_data["json_data"]["stop_price"] = position_stop
                                current_position_data["json_data"]["peak_price"] = position_peak

                                # PnL metrics
                                pnl_pct = (
                                    (exit_price - position_entry_price) / position_entry_price
                                    if position_direction == "long"
                                    else (position_entry_price - exit_price) / position_entry_price
                                )
                                if pnl_pct > 0:
                                    wins.append(pnl_pct)
                                else:
                                    losses.append(abs(pnl_pct))

                                # MFE
                                mfe = (
                                    (position_peak - position_entry_price) / position_entry_price
                                    if position_direction == "long"
                                    else (position_entry_price - position_peak) / position_entry_price
                                )
                                mfes.append(mfe)

                                positions_to_create.append(Position(**current_position_data))
                                current_position_data = None

                            # Reset all perps state
                            hold_lengths.append(bars_since_change)
                            bars_since_change = 0
                            current_config = None
                            position_entry_price = None
                            position_lower_bound = None
                            position_upper_bound = None
                            position_direction = None
                            position_peak = None
                            position_stop = None

                            continue

                    # 2. Update peak using current bar (for NEXT bar's stop)
                    if position_direction == "long":
                        if position_peak is None or bar_high > position_peak:
                            position_peak = bar_high
                    else:
                        if position_peak is None or bar_low < position_peak:
                            position_peak = bar_low

                    # 3. Compute new stop (for NEXT bar)
                    if position_direction == "long":
                        new_stop = position_peak * (1 - trail_pct)
                        if position_stop is None or new_stop > position_stop:
                            position_stop = new_stop
                    else:
                        new_stop = position_peak * (1 + trail_pct)
                        if position_stop is None or new_stop < position_stop:
                            position_stop = new_stop

                # Check if price touches bounds (execution model depends on policy)
                if policy_mode == "perps":
                    # OHLC-based for perps (consistent with trailing stops)
                    touched_lower = bar_low <= position_lower_bound
                    touched_upper = bar_high >= position_upper_bound
                else:
                    # Close-based for LP (backward compatible)
                    touched_lower = close_price <= position_lower_bound
                    touched_upper = close_price >= position_upper_bound

                if touched_lower or touched_upper:
                    total_touches += 1

                    # Determine exit reason (resolve both-touched to adverse side for perps)
                    if touched_lower and touched_upper:
                        # Both bounds hit on same bar
                        if policy_mode == "perps" and position_direction is not None:
                            # Conservative: adverse side (lower for long, upper for short)
                            exit_reason = (
                                ExitReason.TOUCHED_LOWER if position_direction == "long"
                                else ExitReason.TOUCHED_UPPER
                            )
                        else:
                            # LP mode: deterministic lower (existing behavior)
                            exit_reason = ExitReason.TOUCHED_LOWER
                    elif touched_lower:
                        exit_reason = ExitReason.TOUCHED_LOWER
                    else:
                        exit_reason = ExitReason.TOUCHED_UPPER

                    # Close position on touch
                    if current_position_data is not None:
                        # Conservative fill for perps (worse of bound vs close)
                        if policy_mode == "perps" and position_direction is not None:
                            # Use bound matching exit_reason (adverse side if both touched)
                            bound_price = (
                                position_lower_bound if exit_reason == ExitReason.TOUCHED_LOWER
                                else position_upper_bound
                            )
                            # Direction-aware worse fill
                            if position_direction == "long":
                                exit_price = min(bound_price, bar_close)
                            else:
                                exit_price = max(bound_price, bar_close)
                        else:
                            exit_price = close_price

                        current_position_data["exit_timestamp"] = ensure_aware_timestamp(ts)
                        current_position_data["exit_price"] = Decimal(str(exit_price))
                        current_position_data["exit_reason"] = exit_reason
                        current_position_data["bars_held"] = bars_since_change
                        current_position_data["status"] = PositionStatus.CLOSED

                        # Record perps metrics (CRITICAL: use exit_price, not bar_close)
                        if policy_mode == "perps" and position_direction is not None:
                            _record_perps_exit_metrics(
                                position_direction, exit_price, position_entry_price,
                                position_peak, wins, losses, mfes
                            )

                        # Track LP return for this position (LP mode only)
                        if policy_mode == "lp":
                            entry_price = float(current_position_data["entry_price"])
                            exit_price_float = float(current_position_data["exit_price"])
                            ret = (exit_price_float - entry_price) / entry_price

                            exit_reason_str = str(current_position_data["exit_reason"])

                            lp_n_trades += 1
                            lp_sum_ret += ret

                            ret_factor = max(1 + ret, 1e-10)
                            lp_log_compound += np.log(ret_factor)

                            if exit_reason_str not in lp_by_exit:
                                lp_by_exit[exit_reason_str] = {"count": 0, "sum_ret": 0.0}
                            lp_by_exit[exit_reason_str]["count"] += 1
                            lp_by_exit[exit_reason_str]["sum_ret"] += ret

                        # Track directional return for this position (directional mode only)
                        if policy_mode == "directional":
                            entry_price = float(current_position_data["entry_price"])
                            exit_price_float = float(current_position_data["exit_price"])

                            # Retrieve exposure from json_data
                            json_data = current_position_data.get("json_data", {})
                            exposure = json_data.get("exposure", 0.0)

                            # Compute return using exposure formula: ret = e * (P1/P0 - 1)
                            price_return = exit_price_float / entry_price - 1
                            ret = exposure * price_return

                            exit_reason_str = str(current_position_data["exit_reason"])

                            dir_n_trades += 1
                            dir_sum_ret += ret

                            ret_factor = max(1 + ret, 1e-10)
                            dir_log_compound += np.log(ret_factor)

                            if exit_reason_str not in dir_by_exit:
                                dir_by_exit[exit_reason_str] = {"count": 0, "sum_ret": 0.0}
                            dir_by_exit[exit_reason_str]["count"] += 1
                            dir_by_exit[exit_reason_str]["sum_ret"] += ret

                        positions_to_create.append(Position(**current_position_data))
                        current_position_data = None

                    # CRITICAL: Reset ALL state including perps state
                    hold_lengths.append(bars_since_change)
                    bars_since_change = 0
                    current_config = None
                    position_entry_price = None
                    position_lower_bound = None
                    position_upper_bound = None
                    if policy_mode == "perps":
                        position_direction = None
                        position_peak = None
                        position_stop = None
                else:
                    bars_in_range += 1

        # Skip config selection if already in perps position
        if policy_mode == "perps" and current_config is not None:
            # In perps mode: lock position, no recomputation
            pass
        elif current_config is None or (
            policy_mode in ["lp", "directional"]
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
                models_dict=models_dict,
                feature_cols=feature_cols,
                touch_tolerance=touch_tolerance,
                horizons=horizons,
                policy_mode=policy_mode,
                conf_threshold=conf_threshold,
                move_threshold=move_threshold,
                perps_widths=perps_widths,
                exposures=exposures,
                skew_threshold=skew_threshold,
                max_abs_exposure=max_abs_exposure,
            )

            if best_config is None:
                no_config_count += 1
                continue

            # Check if we should change position
            should_change = current_config is None or _config_significantly_different(
                current_config, best_config
            )

            if should_change:
                # Close existing position if any (rebalance)
                if current_config is not None and current_position_data is not None:
                    current_position_data["exit_timestamp"] = ensure_aware_timestamp(ts)
                    current_position_data["exit_price"] = Decimal(str(close_price))
                    current_position_data["exit_reason"] = ExitReason.REBALANCED
                    current_position_data["bars_held"] = bars_since_change
                    current_position_data["status"] = PositionStatus.CLOSED

                    # Track LP return for this position (LP mode only)
                    if policy_mode == "lp":
                        entry_price = float(current_position_data["entry_price"])
                        exit_price = float(current_position_data["exit_price"])
                        ret = (exit_price - entry_price) / entry_price

                        exit_reason = str(current_position_data["exit_reason"])

                        lp_n_trades += 1
                        lp_sum_ret += ret

                        ret_factor = max(1 + ret, 1e-10)
                        lp_log_compound += np.log(ret_factor)

                        if exit_reason not in lp_by_exit:
                            lp_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
                        lp_by_exit[exit_reason]["count"] += 1
                        lp_by_exit[exit_reason]["sum_ret"] += ret

                    # Track directional return for this position (directional mode only)
                    if policy_mode == "directional":
                        entry_price = float(current_position_data["entry_price"])
                        exit_price = float(current_position_data["exit_price"])

                        # Retrieve exposure from json_data
                        json_data = current_position_data.get("json_data", {})
                        exposure = json_data.get("exposure", 0.0)

                        # Compute return using exposure formula: ret = e * (P1/P0 - 1)
                        price_return = exit_price / entry_price - 1
                        ret = exposure * price_return

                        exit_reason = str(current_position_data["exit_reason"])

                        dir_n_trades += 1
                        dir_sum_ret += ret

                        ret_factor = max(1 + ret, 1e-10)
                        dir_log_compound += np.log(ret_factor)

                        if exit_reason not in dir_by_exit:
                            dir_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
                        dir_by_exit[exit_reason]["count"] += 1
                        dir_by_exit[exit_reason]["sum_ret"] += ret

                    positions_to_create.append(Position(**current_position_data))
                    current_position_data = None

                    hold_lengths.append(bars_since_change)
                    rebalance_count += 1
                    bars_since_change = 0

                # Open new position
                current_config = best_config
                position_entry_price = close_price
                position_lower_bound = position_entry_price * (
                    1 + best_config.lower_pct
                )
                position_upper_bound = position_entry_price * (
                    1 + best_config.upper_pct
                )

                # Create new position data
                json_data_dict = {
                    "p_touch_lower": best_config.p_touch_lower,
                    "p_touch_upper": best_config.p_touch_upper,
                    "width": best_config.width,
                    "asymmetry": best_config.asymmetry,
                    "directional_skew": best_config.p_touch_upper - best_config.p_touch_lower,
                    "decision_horizon": min(horizons),
                    "p_exit": best_config.p_touch_upper + best_config.p_touch_lower,
                    "policy_mode": policy_mode,
                }
                if model_cutoff is not None:
                    json_data_dict["model_cutoff"] = str(model_cutoff)

                # Initialize perps position state
                if policy_mode == "perps":
                    position_direction = best_config.direction
                    position_peak = bar_close
                    position_stop = (
                        bar_close * (1 - trail_pct)
                        if position_direction == "long"
                        else bar_close * (1 + trail_pct)
                    )

                    json_data_dict["direction"] = position_direction
                    json_data_dict["initial_stop"] = position_stop
                    json_data_dict["trail_pct"] = trail_pct

                # Store exposure for directional mode
                if policy_mode == "directional":
                    json_data_dict["exposure"] = best_config.exposure

                current_position_data = {
                    "symbol": symbol,
                    "ml_config": config,
                    "position_type": PositionType.BACKTEST,
                    "lower_bound": best_config.lower_pct,
                    "upper_bound": best_config.upper_pct,
                    "borrow_ratio": best_config.borrow_ratio,
                    "entry_timestamp": ensure_aware_timestamp(ts),
                    "entry_price": Decimal(str(close_price)),
                    "status": PositionStatus.OPEN,
                    "json_data": json_data_dict,
                }

    # Close remaining open position
    if current_position_data is not None:
        current_position_data["exit_timestamp"] = ensure_aware_timestamp(ts)
        current_position_data["exit_price"] = Decimal(str(bar_close))
        current_position_data["exit_reason"] = ExitReason.MAX_DURATION
        current_position_data["bars_held"] = bars_since_change
        current_position_data["status"] = PositionStatus.CLOSED

        # Record perps metrics (using bar_close as mark/exit assumption)
        if policy_mode == "perps" and position_direction is not None:
            _record_perps_exit_metrics(
                position_direction, bar_close, position_entry_price,
                position_peak, wins, losses, mfes
            )

        # Track LP return for this position (LP mode only)
        if policy_mode == "lp":
            entry_price = float(current_position_data["entry_price"])
            exit_price = float(current_position_data["exit_price"])
            ret = (exit_price - entry_price) / entry_price

            exit_reason = str(current_position_data["exit_reason"])

            lp_n_trades += 1
            lp_sum_ret += ret

            ret_factor = max(1 + ret, 1e-10)
            lp_log_compound += np.log(ret_factor)

            if exit_reason not in lp_by_exit:
                lp_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
            lp_by_exit[exit_reason]["count"] += 1
            lp_by_exit[exit_reason]["sum_ret"] += ret

        # Track directional return for this position (directional mode only)
        if policy_mode == "directional":
            entry_price = float(current_position_data["entry_price"])
            exit_price = float(current_position_data["exit_price"])

            # Retrieve exposure from json_data
            json_data = current_position_data.get("json_data", {})
            exposure = json_data.get("exposure", 0.0)

            # Compute return using exposure formula: ret = e * (P1/P0 - 1)
            price_return = exit_price / entry_price - 1
            ret = exposure * price_return

            exit_reason = str(current_position_data["exit_reason"])

            dir_n_trades += 1
            dir_sum_ret += ret

            ret_factor = max(1 + ret, 1e-10)
            dir_log_compound += np.log(ret_factor)

            if exit_reason not in dir_by_exit:
                dir_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
            dir_by_exit[exit_reason]["count"] += 1
            dir_by_exit[exit_reason]["sum_ret"] += ret

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

    # Perps metrics
    perps_metrics = None
    if policy_mode == "perps":
        # Calculate counts and sums for pooling
        n_wins = len(wins)
        n_losses = len(losses)
        sum_wins = sum(wins) if wins else 0.0
        sum_losses = sum(losses) if losses else 0.0

        # Calculate log-compound return for this window
        # log_compound = Σlog(1+r) allows efficient pooling across windows
        # Use log1p for numerical stability
        log_compound = 0.0
        for w in wins:
            log_compound += np.log1p(w)  # log1p(w) = log(1+w), more accurate for small w
        for loss in losses:
            # Guard against 100% loss (1-loss <= 0) by clipping
            loss_factor = max(1 - loss, 1e-10)  # Clip to prevent log(0)
            log_compound += np.log(loss_factor)  # log(1-loss) for losses

        # Per-window metrics (for debugging, not aggregated)
        win_rate = n_wins / (n_wins + n_losses) if (n_wins + n_losses) > 0 else 0.0
        avg_win = sum_wins / n_wins if n_wins > 0 else 0.0
        avg_loss = sum_losses / n_losses if n_losses > 0 else 0.0
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0
        mfe_avg = np.mean(mfes) if mfes else 0.0

        perps_metrics = {
            # Poolable aggregates
            "n_wins": n_wins,
            "n_losses": n_losses,
            "sum_wins": sum_wins,
            "sum_losses": sum_losses,
            "log_compound": log_compound,
            "n_trailing_stops": n_trailing_stops,
            # Per-window metrics (keep for debugging)
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "mfe_avg": mfe_avg,
        }

    # LP metrics
    lp_metrics = None
    if policy_mode == "lp":
        compound_ret = np.exp(lp_log_compound) - 1.0 if lp_n_trades > 0 else 0.0
        avg_ret = lp_sum_ret / lp_n_trades if lp_n_trades > 0 else 0.0

        lp_metrics = {
            "n_trades": lp_n_trades,
            "sum_ret": lp_sum_ret,
            "log_compound": lp_log_compound,
            "compound_ret": compound_ret,
            "avg_ret": avg_ret,
            "by_exit": lp_by_exit,
        }

    # Directional metrics
    directional_metrics = None
    if policy_mode == "directional":
        compound_ret = np.exp(dir_log_compound) - 1.0 if dir_n_trades > 0 else 0.0
        avg_ret = dir_sum_ret / dir_n_trades if dir_n_trades > 0 else 0.0

        directional_metrics = {
            "n_trades": dir_n_trades,
            "sum_ret": dir_sum_ret,
            "log_compound": dir_log_compound,
            "compound_ret": compound_ret,
            "avg_ret": avg_ret,
            "by_exit": dir_by_exit,
        }

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
        bars_with_no_valid_config=no_config_count,
        perps_metrics=perps_metrics,
        lp_metrics=lp_metrics,
        directional_metrics=directional_metrics,
    )


def _find_optimal_config(
    bar_rows: pd.DataFrame,
    models_dict: dict[str, Any],
    feature_cols: list[str],
    touch_tolerance: float,
    horizons: list[int],
    policy_mode: str = "lp",
    conf_threshold: float | None = None,
    move_threshold: float | None = None,
    perps_widths: list[float] | None = None,
    exposures: list[float] | None = None,
    skew_threshold: float | None = None,
    max_abs_exposure: float | None = None,
) -> LPConfig | None:
    """Find optimal config using competing-risks models."""
    # Extract widths and asymmetries from bar_rows
    widths = sorted(bar_rows["width"].unique()) if "width" in bar_rows.columns else None
    asymmetries = (
        sorted(bar_rows["asymmetry"].unique())
        if "asymmetry" in bar_rows.columns
        else None
    )

    # Get base features (all configs have same features, differ only in bounds)
    # Filter out config-only cols and *_missing cols (generated by prepare_features at inference time)
    base_features = bar_rows.iloc[0:1][
        [
            c
            for c in feature_cols
            if c not in {"k", "width", "asymmetry", "dist_to_lower_pct", "dist_to_upper_pct"}
            and not c.endswith("_missing")
        ]
    ]

    # Use competing-risks models to select config by P(TIMEOUT@decision_horizon)
    from quant_core.prediction import predict_competing_risks_multi_horizon

    # Use shortest horizon for config selection (want configs that survive near-term)
    decision_horizon = min(horizons)

    configs = [(w, a) for w in widths for a in asymmetries]

    # Filter to perps subset if in perps mode
    if policy_mode == "perps":
        # Safety check: perps_widths must be provided
        if perps_widths is None:
            logger.error("perps_widths is None in perps mode - cannot filter configs")
            return None

        # Robust float comparison using np.isclose to avoid 0.01 vs 0.0100000000002 issues
        configs = [
            (w, a) for (w, a) in configs
            if any(np.isclose(w, pw) for pw in perps_widths) and np.isclose(a, 0.0)
        ]
        if not configs:
            logger.warning(
                f"No perps configs (symmetric, width in perps_widths) in training grid. "
                f"Training grid widths: {widths}, asymmetries: {asymmetries}. "
                f"Perps widths: {perps_widths}. "
                f"Training grid must include perps widths and asymmetry=0.0"
            )
            return None

    valid_configs = []

    for width, asymmetry in configs:
        lower_pct = -width * (0.5 - asymmetry)
        upper_pct = width * (0.5 + asymmetry)

        preds = predict_competing_risks_multi_horizon(
            models_dict, base_features, horizons,
            lower_pct, upper_pct,
            width, asymmetry,
            feature_cols
        )

        # Extract probabilities at decision horizon
        p_touch_lower = preds[decision_horizon]["DOWN_FIRST"]
        p_touch_upper = preds[decision_horizon]["UP_FIRST"]
        p_timeout = preds[decision_horizon]["TIMEOUT"]

        if policy_mode == "perps":
            # Perps: filter by directional confidence
            conf = max(p_touch_upper, p_touch_lower)
            move_prob = p_touch_upper + p_touch_lower

            if conf >= conf_threshold and move_prob >= move_threshold:
                direction = "long" if p_touch_upper > p_touch_lower else "short"
                valid_configs.append({
                    "width": width,
                    "asymmetry": asymmetry,
                    "lower_pct": lower_pct,
                    "upper_pct": upper_pct,
                    "p_touch_lower": p_touch_lower,
                    "p_touch_upper": p_touch_upper,
                    "p_timeout": p_timeout,
                    "conf": conf,
                    "direction": direction,
                })
        else:
            # LP: existing touch_tolerance logic
            p_exit = p_touch_lower + p_touch_upper
            if p_exit <= touch_tolerance:
                valid_configs.append({
                    "width": width,
                    "asymmetry": asymmetry,
                    "lower_pct": lower_pct,
                    "upper_pct": upper_pct,
                    "p_touch_lower": p_touch_lower,
                    "p_touch_upper": p_touch_upper,
                    "p_timeout": p_timeout,
                })

    # If no valid configs, return None
    if not valid_configs:
        return None

    # Select best config
    if policy_mode == "perps":
        best_config = max(valid_configs, key=lambda x: x["conf"])
    else:
        best_config = max(valid_configs, key=lambda x: x["p_timeout"])

    # Create LPConfig
    lp_config = LPConfig(
        lower_pct=best_config["lower_pct"],
        upper_pct=best_config["upper_pct"],
        borrow_ratio=0.5 + best_config["asymmetry"],
        p_touch_lower=best_config["p_touch_lower"],
        p_touch_upper=best_config["p_touch_upper"],
        width=best_config["width"],
        asymmetry=best_config["asymmetry"],
        direction=best_config.get("direction"),
    )

    # Choose exposure based on directional skew (directional mode only)
    if policy_mode == "directional" and exposures is not None:
        # Calculate directional skew
        directional_skew = best_config["p_touch_upper"] - best_config["p_touch_lower"]

        # Simple threshold-based mapping
        if abs(directional_skew) < skew_threshold:
            # Low confidence: zero exposure (neutral)
            lp_config.exposure = 0.0
        else:
            # High confidence: full exposure in predicted direction
            if directional_skew > 0:
                # Bullish: long
                lp_config.exposure = max_abs_exposure
            else:
                # Bearish: short
                lp_config.exposure = -max_abs_exposure
    else:
        # LP or perps mode: no exposure
        lp_config.exposure = 0.0

    return lp_config


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
    retrain_cadence_days: int | None = None,
    train_window_days: int | None = None,
    holdout_days: int | None = None,
    policy_mode: str | None = None,
    **backtest_kwargs,
) -> dict:
    """Run walk-forward simulation with rolling retrains.

    Args:
        config: MLConfig to simulate
        retrain_cadence_days: Days between retrains (default from config)
        train_window_days: Days of data to train on (default from config)
        holdout_days: Optional final holdout period (default from config)
        policy_mode: Policy mode ("lp" or "perps", default from config)
        **backtest_kwargs: Additional args passed to backtest

    Returns:
        Dict with aggregated metrics and per-slice results
    """
    # Get simulation params from config with optional overrides
    sim_params = config.get_simulation_params()

    cadence = (
        retrain_cadence_days
        if retrain_cadence_days is not None
        else sim_params["retrain_cadence_days"]
    )
    window = (
        train_window_days
        if train_window_days is not None
        else sim_params["train_window_days"]
    )
    holdout = holdout_days if holdout_days is not None else sim_params["holdout_days"]
    policy = policy_mode if policy_mode is not None else sim_params["policy_mode"]

    # Get date range from CandleData without loading all data
    from quant_tick.models import CandleData

    candle_bounds = CandleData.objects.filter(candle=config.candle)
    if not candle_bounds.exists():
        logger.error(f"{config}: No CandleData found")
        return {}

    min_ts = candle_bounds.order_by("timestamp").first().timestamp
    max_ts = candle_bounds.order_by("-timestamp").first().timestamp
    logger.info(f"{config}: CandleData range {min_ts} to {max_ts}")

    # Reserve holdout if specified
    if holdout:
        max_ts = max_ts - timedelta(days=holdout)
        logger.info(f"{config}: reserved {holdout} days as final holdout")

    # Generate cutoff dates
    train_window_delta = timedelta(days=window)
    cadence_delta = timedelta(days=cadence)

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
    total_no_config_bars = 0

    for i, cutoff in enumerate(tqdm(cutoffs, desc="Walk-forward windows")):
        logger.debug(f"{config}: window {i+1}/{len(cutoffs)} - cutoff={cutoff}")

        # Load training data for this window only
        train_start = cutoff - train_window_delta

        try:
            train_df = load_training_df(config, train_start, cutoff)
        except ValueError as e:
            logger.warning(f"{config}: {e}")
            windows_skipped += 1
            skip_reasons["no_training_data"] = (
                skip_reasons.get("no_training_data", 0) + 1
            )
            continue

        if len(train_df) == 0:
            logger.warning(f"{config}: no training data for cutoff {cutoff}")
            windows_skipped += 1
            skip_reasons["no_training_data"] = (
                skip_reasons.get("no_training_data", 0) + 1
            )
            continue

        # Validate competing-risks schema structure
        # NOTE: Old hazard schema validation (with k column) is obsolete.
        # Competing-risks models use bars × configs structure without k dimension.
        if (
            "bar_idx" in train_df.columns
            and "config_id" in train_df.columns
        ):
            is_valid, error = MLSchema.validate_bar_config_invariants(train_df)
            if not is_valid:
                logger.error(f"{config}: Training window invalid - {error}, skipping")
                windows_skipped += 1
                skip_reasons["schema_invalid"] = (
                    skip_reasons.get("schema_invalid", 0) + 1
                )
                del train_df
                continue

        # Train models on window
        try:
            models_dict, feature_cols, cv_metrics, holdout_metrics = train_core(
                df=train_df,
                horizons=config.get_horizons(),
                n_splits=0,  # Skip CV in walk-forward (still keeps calib/test holdout)
                embargo_bars=96,
                holdout_pct=0.15,  # Smaller holdout for walk-forward
                optuna_n_trials=0,  # Never run Optuna in walk-forward (too slow)
            )
        except Exception as e:
            logger.error(f"{config}: training failed for cutoff {cutoff}: {e}")
            windows_skipped += 1
            skip_reasons["training_failed"] = skip_reasons.get("training_failed", 0) + 1
            continue

        # Validate competing-risks bundle structure
        horizons = config.get_horizons()
        expected_keys = {f"first_hit_h{H}" for H in horizons}
        missing_keys = expected_keys - set(models_dict.keys())

        if missing_keys:
            logger.error(f"{config}: missing competing-risks models: {missing_keys}")
            windows_skipped += 1
            skip_reasons["missing_models"] = skip_reasons.get("missing_models", 0) + 1
            continue

        # No extraction needed - pass full models_dict to backtest

        # Save train size before freeing memory
        train_size = len(train_df)
        del train_df

        # Load scoring data for next cadence period
        score_start = cutoff
        score_end = cutoff + cadence_delta

        try:
            score_df = load_training_df(config, score_start, score_end)
        except ValueError as e:
            logger.warning(f"{config}: {e}")
            windows_skipped += 1
            skip_reasons["no_scoring_data"] = skip_reasons.get("no_scoring_data", 0) + 1
            continue

        if len(score_df) == 0:
            logger.warning(f"{config}: no scoring data for cutoff {cutoff}")
            windows_skipped += 1
            skip_reasons["no_scoring_data"] = skip_reasons.get("no_scoring_data", 0) + 1
            del score_df
            continue

        # Validate scoring window hazard schema
        if (
            "bar_idx" in score_df.columns
            and "config_id" in score_df.columns
        ):
            is_valid, error = MLSchema.validate_bar_config_invariants(score_df)
            if not is_valid:
                logger.warning(f"Window {cutoff}: scoring data invalid - {error}")
                windows_skipped += 1
                skip_reasons["scoring_schema_invalid"] = (
                    skip_reasons.get("scoring_schema_invalid", 0) + 1
                )
                del score_df
                continue

        # Validate features match using centralized schema
        # Use centralized schema to get data features (excludes config cols added during prediction)
        data_feature_cols = MLSchema.get_data_features(feature_cols)
        available_cols = set(score_df.columns)
        missing_cols = [c for c in data_feature_cols if c not in available_cols]
        if missing_cols:
            logger.error(
                f"{config}: Cannot run backtest - {len(missing_cols)} required feature(s) missing: "
                f"{', '.join(missing_cols[:10])}{'...' if len(missing_cols) > 10 else ''}. "
                f"Skipping this window to prevent schema mismatch."
            )
            windows_skipped += 1
            skip_reasons["missing_features"] = (
                skip_reasons.get("missing_features", 0) + 1
            )

            # Track which features are chronic offenders
            for col in missing_cols:
                missing_feature_counts[col] = missing_feature_counts.get(col, 0) + 1
            continue

        # Run backtest on scoring slice with trained models (don't clear positions from previous slices)
        backtest_result = _run_backtest(
            config=config,
            models_dict=models_dict,
            feature_cols=feature_cols,
            df=score_df,
            model_cutoff=cutoff,
            clear_positions=False,
            policy_mode=policy,
        )

        # Free scoring data memory
        score_size = len(score_df)
        del score_df

        if backtest_result:
            total_no_config_bars += backtest_result.bars_with_no_valid_config
            slice_results.append(
                {
                    "cutoff": cutoff,
                    "train_size": train_size,
                    "score_size": score_size,
                    "cv_avg_logloss": cv_metrics.get("avg_logloss", 0.0),
                    "holdout_avg_logloss": holdout_metrics.get("avg_logloss", 0.0),
                    "backtest_touch_rate": backtest_result.touch_rate,
                    "backtest_pct_in_range": backtest_result.pct_in_range,
                    "backtest_rebalances": backtest_result.rebalances,
                    "backtest_positions": backtest_result.positions_created,
                    "bars_with_no_valid_config": backtest_result.bars_with_no_valid_config,
                    "perps_metrics": backtest_result.perps_metrics,
                    "lp_metrics": backtest_result.lp_metrics,
                    "directional_metrics": backtest_result.directional_metrics,
                }
            )

            logger.info(
                f"{config}: window {i+1} - CV logloss: {cv_metrics.get('avg_logloss', 0.0):.4f}, "
                f"holdout logloss: {holdout_metrics.get('avg_logloss', 0.0):.4f}, "
                f"touch_rate: {backtest_result.touch_rate:.2%}"
            )

    # Aggregate metrics
    if slice_results:
        agg_metrics = {
            "n_windows": len(slice_results),
            "avg_cv_logloss": sum(r["cv_avg_logloss"] for r in slice_results)
            / len(slice_results),
            "avg_holdout_logloss": sum(r["holdout_avg_logloss"] for r in slice_results)
            / len(slice_results),
            "avg_touch_rate": sum(r["backtest_touch_rate"] for r in slice_results)
            / len(slice_results),
            "avg_pct_in_range": sum(r["backtest_pct_in_range"] for r in slice_results)
            / len(slice_results),
            "total_rebalances": sum(r["backtest_rebalances"] for r in slice_results),
            "total_positions": sum(r["backtest_positions"] for r in slice_results),
        }

        # Aggregate perps metrics if present
        perps_slices = [r for r in slice_results if r.get("perps_metrics")]
        if perps_slices:
            # Pool aggregates across all windows
            total_n_wins = 0
            total_n_losses = 0
            total_sum_wins = 0.0
            total_sum_losses = 0.0
            total_log_compound = 0.0
            total_trailing_stops = 0

            for r in perps_slices:
                pm = r["perps_metrics"]
                total_n_wins += pm["n_wins"]
                total_n_losses += pm["n_losses"]
                total_sum_wins += pm["sum_wins"]
                total_sum_losses += pm["sum_losses"]
                total_log_compound += pm["log_compound"]
                total_trailing_stops += pm["n_trailing_stops"]

            # Calculate pooled metrics from aggregates
            total_trades = total_n_wins + total_n_losses

            if total_trades > 0:
                pooled_win_rate = total_n_wins / total_trades
                pooled_avg_win = total_sum_wins / total_n_wins if total_n_wins > 0 else 0.0
                pooled_avg_loss = total_sum_losses / total_n_losses if total_n_losses > 0 else 0.0
                pooled_profit_factor = total_sum_wins / total_sum_losses if total_sum_losses > 0 else 0.0
            else:
                pooled_win_rate = 0.0
                pooled_avg_win = 0.0
                pooled_avg_loss = 0.0
                pooled_profit_factor = 0.0

            # Calculate return metrics
            sum_ret = total_sum_wins - total_sum_losses
            compound_ret = np.exp(total_log_compound) - 1.0 if total_trades > 0 else 0.0

            agg_metrics["perps_metrics"] = {
                "n_trades": total_trades,
                "total_trailing_stops": total_trailing_stops,
                "win_rate": pooled_win_rate,
                "profit_factor": pooled_profit_factor,
                "avg_win": pooled_avg_win,
                "avg_loss": pooled_avg_loss,
                "sum_ret": sum_ret,
                "compound_ret": compound_ret,
                # Keep MFE averaging for now (not critical)
                "avg_mfe": sum(r["perps_metrics"]["mfe_avg"] for r in perps_slices) / len(perps_slices)
                           if perps_slices else 0.0,
            }

        # Aggregate LP metrics if present
        lp_slices = [r for r in slice_results if r.get("lp_metrics")]
        if lp_slices:
            # Pool aggregates across all windows
            total_lp_n_trades = 0
            total_lp_sum_ret = 0.0
            total_lp_log_compound = 0.0
            pooled_lp_by_exit = {}

            for r in lp_slices:
                lm = r["lp_metrics"]
                total_lp_n_trades += lm["n_trades"]
                total_lp_sum_ret += lm["sum_ret"]
                total_lp_log_compound += lm["log_compound"]

                # Merge by_exit breakdowns
                for exit_reason, data in lm.get("by_exit", {}).items():
                    if exit_reason not in pooled_lp_by_exit:
                        pooled_lp_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
                    pooled_lp_by_exit[exit_reason]["count"] += data["count"]
                    pooled_lp_by_exit[exit_reason]["sum_ret"] += data["sum_ret"]

            # Calculate pooled metrics
            pooled_lp_compound_ret = np.exp(total_lp_log_compound) - 1.0 if total_lp_n_trades > 0 else 0.0
            pooled_lp_avg_ret = total_lp_sum_ret / total_lp_n_trades if total_lp_n_trades > 0 else 0.0

            agg_metrics["lp_metrics"] = {
                "n_trades": total_lp_n_trades,
                "sum_ret": total_lp_sum_ret,
                "compound_ret": pooled_lp_compound_ret,
                "avg_ret": pooled_lp_avg_ret,
                "by_exit": pooled_lp_by_exit,
            }

        # Aggregate directional metrics if present
        dir_slices = [r for r in slice_results if r.get("directional_metrics")]
        if dir_slices:
            # Pool aggregates across all windows
            total_dir_n_trades = 0
            total_dir_sum_ret = 0.0
            total_dir_log_compound = 0.0
            pooled_dir_by_exit = {}

            for r in dir_slices:
                dm = r["directional_metrics"]
                total_dir_n_trades += dm["n_trades"]
                total_dir_sum_ret += dm["sum_ret"]
                total_dir_log_compound += dm["log_compound"]

                # Merge by_exit breakdowns
                for exit_reason, data in dm.get("by_exit", {}).items():
                    if exit_reason not in pooled_dir_by_exit:
                        pooled_dir_by_exit[exit_reason] = {"count": 0, "sum_ret": 0.0}
                    pooled_dir_by_exit[exit_reason]["count"] += data["count"]
                    pooled_dir_by_exit[exit_reason]["sum_ret"] += data["sum_ret"]

            # Calculate pooled metrics
            pooled_dir_compound_ret = np.exp(total_dir_log_compound) - 1.0 if total_dir_n_trades > 0 else 0.0
            pooled_dir_avg_ret = total_dir_sum_ret / total_dir_n_trades if total_dir_n_trades > 0 else 0.0

            agg_metrics["directional_metrics"] = {
                "n_trades": total_dir_n_trades,
                "sum_ret": total_dir_sum_ret,
                "compound_ret": pooled_dir_compound_ret,
                "avg_ret": pooled_dir_avg_ret,
                "by_exit": pooled_dir_by_exit,
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
                for feat, count in sorted(
                    chronic_features.items(), key=lambda x: -x[1]
                ):
                    pct = count / windows_attempted
                    logger.warning(
                        f"  - {feat}: missing in {count}/{windows_attempted} windows ({pct:.1%})"
                    )

    return WalkForwardResult(
        aggregate_metrics=agg_metrics,
        slice_results=slice_results,
        windows_attempted=windows_attempted,
        windows_skipped=windows_skipped,
        skip_reasons=skip_reasons,
        chronic_missing_features=missing_feature_counts,
        bars_with_no_valid_config=total_no_config_bars,
    )
