import logging
from datetime import timedelta
from decimal import Decimal
from typing import Any

import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from sklearn.ensemble import RandomForestClassifier

from quant_tick.constants import ExitReason, PositionStatus, PositionType
from quant_tick.lib.ml import compute_ewma_volatility, generate_signals
from quant_tick.lib.rf_interpretation import (
    compute_tree_contributions,
    get_feature_contribution_summary,
)
from quant_tick.lib.trend_scanning import (
    generate_trend_windows,
    rank_trends,
    scan_trends,
)
from quant_tick.models import (
    MLArtifact,
    MLConfig,
    MLFeatureData,
    MLRun,
    Position,
    TrendScan,
)

logger = logging.getLogger(__name__)


def _compute_trade_exit(
    entry_idx: int,
    entry_price: Decimal,
    direction: int,
    prices: np.ndarray,
    timestamps: np.ndarray,
    highs: np.ndarray | None,
    lows: np.ndarray | None,
    tp_price: Decimal | None,
    sl_price: Decimal | None,
    max_holding_bars: int,
    costs_bps: int,
) -> dict:
    """Compute trade exit using triple-barrier logic.

    Shared helper for simulate_trades implementations. Contains all exit detection
    and PnL calculation logic.

    Args:
        entry_idx: Index of entry bar
        entry_price: Entry price (Decimal)
        direction: 1 for long, -1 for short
        prices: Array of close prices
        timestamps: Array of timestamps
        highs: Array of high prices (or None)
        lows: Array of low prices (or None)
        tp_price: Take profit price (Decimal)
        sl_price: Stop loss price (Decimal)
        max_holding_bars: Maximum holding period
        costs_bps: Transaction costs in basis points

    Returns:
        Tuple of (exit_idx, exit_price, exit_reason, exit_timestamp, pnl)
    """
    exit_idx = None
    exit_price = None
    exit_reason = None
    bars_held = 0

    for j in range(entry_idx, min(entry_idx + max_holding_bars, len(prices))):
        bars_held += 1

        high = Decimal(str(highs[j])) if highs is not None else Decimal(str(prices[j]))
        low = Decimal(str(lows[j])) if lows is not None else Decimal(str(prices[j]))
        close = Decimal(str(prices[j]))

        if direction == 1:
            if high >= tp_price:
                exit_idx = j
                exit_price = tp_price
                exit_reason = ExitReason.TAKE_PROFIT
                break
            if low <= sl_price:
                exit_idx = j
                exit_price = sl_price
                exit_reason = ExitReason.STOP_LOSS
                break
        else:
            if low <= tp_price:
                exit_idx = j
                exit_price = tp_price
                exit_reason = ExitReason.TAKE_PROFIT
                break
            if high >= sl_price:
                exit_idx = j
                exit_price = sl_price
                exit_reason = ExitReason.STOP_LOSS
                break

        if bars_held >= max_holding_bars:
            exit_idx = j
            exit_price = close
            exit_reason = ExitReason.MAX_DURATION
            break

    if exit_idx is None:
        exit_idx = len(prices) - 1
        exit_price = Decimal(str(prices[exit_idx]))
        exit_reason = ExitReason.MAX_DURATION

    exit_timestamp = pd.Timestamp(timestamps[exit_idx]).to_pydatetime()

    cost = Decimal(str(costs_bps / 10000))

    if direction == 1:
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price

    pnl = pnl - (cost * 2)

    return exit_idx, exit_price, exit_reason, exit_timestamp, pnl


def simulate_trades(
    signals: pd.DataFrame,
    prices: np.ndarray,
    timestamps: np.ndarray,
    ml_run: Any,
    costs_bps: int = 5,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding_bars: int = 48,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volatilities: np.ndarray | None = None,
    starting_equity: Decimal | None = None,
    meta_used: bool = False,
    feature_contributions: list[dict] | None = None,
) -> tuple[list, Decimal]:
    """Simulate trade execution with triple-barrier exits and persist Position records.

    Entry: next bar open after signal
    Exit: TP/SL on intrabar high/low or max_holding_bars
    TP/SL are volatility-scaled matching triple-barrier labeling.

    Args:
        feature_contributions: Optional list of per-signal feature contribution dicts.
            Should align with non-zero signals. Stored in Position.json_data as
            {"feature_contributions": {"feature_name": contribution, ...}}.

    Note:
        Refactored to use bulk_create() for 10-50× speedup.
        Builds Position objects in-memory, then batch-inserts at end.
    """
    trades = []
    positions_to_create = []
    equity = starting_equity if starting_equity is not None else Decimal("1.0")
    symbol = ml_run.ml_config.symbol

    pt_mult_decimal = Decimal(str(pt_mult))
    sl_mult_decimal = Decimal(str(sl_mult))

    contribution_idx = 0

    for i, signal in enumerate(signals):
        if signal == 0:
            continue

        if i + 1 >= len(prices):
            continue

        entry_idx = i + 1
        entry_price = Decimal(str(prices[entry_idx]))
        entry_timestamp = pd.Timestamp(timestamps[entry_idx]).to_pydatetime()
        direction = signal

        volatility = Decimal(str(volatilities[entry_idx])) if volatilities is not None else Decimal("0.01")

        if direction == 1:
            tp_price = entry_price * (Decimal("1") + pt_mult_decimal * volatility)
            sl_price = entry_price * (Decimal("1") - sl_mult_decimal * volatility)
        else:
            tp_price = entry_price * (Decimal("1") - pt_mult_decimal * volatility)
            sl_price = entry_price * (Decimal("1") + sl_mult_decimal * volatility)

        max_duration = entry_timestamp + timedelta(hours=max_holding_bars)

        exit_idx, exit_price, exit_reason, exit_timestamp, pnl = _compute_trade_exit(
            entry_idx, entry_price, direction, prices, timestamps,
            highs, lows, tp_price, sl_price, max_holding_bars, costs_bps
        )

        equity *= (Decimal("1.0") + pnl)

        json_data = None
        if feature_contributions and contribution_idx < len(feature_contributions):
            json_data = {"feature_contributions": feature_contributions[contribution_idx]}
            contribution_idx += 1

        position = Position(
            ml_run=ml_run,
            ml_signal=None,
            symbol=symbol,
            position_type=PositionType.BACKTEST,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            take_profit=tp_price,
            stop_loss=sl_price,
            max_duration=max_duration,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            side=direction,
            size=Decimal("1.0"),
            status=PositionStatus.CLOSED,
            json_data=json_data,
        )
        positions_to_create.append(position)

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "direction": direction,
            "pnl": float(pnl),
            "equity": float(equity),
            "position_id": None,
            "meta_filtered": meta_used
        })

    if positions_to_create:
        created_positions = Position.objects.bulk_create(positions_to_create, batch_size=1000)
        for i, position in enumerate(created_positions):
            trades[i]["position_id"] = position.id

    return trades, equity


def compute_metrics(trades: list, equity: Decimal) -> dict:
    """Compute performance metrics.

    Args:
        trades: List of trade dicts with pnl and equity
        equity: Current equity level (Decimal)
    """
    if not trades:
        return {
            "total_trades": 0,
            "final_equity": float(equity),
            "total_return": float(equity - Decimal("1.0")),
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }

    returns = [t["pnl"] for t in trades]

    total_return = float(equity - Decimal("1.0"))
    avg_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 0.01
    sharpe = avg_return / std_return if std_return > 0 else 0

    equity_curve = [t["equity"] for t in trades]

    if returns and (1.0 + returns[0]) != 0:
        starting_equity = equity_curve[0] / (1.0 + returns[0])
    else:
        starting_equity = float(equity)

    full_curve = [starting_equity] + equity_curve

    peak = full_curve[0]
    max_dd = 0

    for eq in full_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns) if returns else 0

    return {
        "total_trades": len(trades),
        "final_equity": float(equity),
        "total_return": total_return,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate)
    }


class Command(BaseCommand):
    r"""Walk-forward backtest with rolling train/test windows.

    This command simulates realistic trading by training on historical data and testing
    on future data, then rolling the window forward. This prevents lookahead bias that
    would occur if you trained on all data at once.

    How it works:
    1. Define train window (e.g., last 5000 bars) and test window (e.g., next 500 bars)
    2. Train model on train window, generate signals on test window
    3. Simulate trades with triple-barrier exits (TP/SL/time limit)
    4. Step forward by step_bars and repeat until end of data
    5. Track cumulative equity and performance metrics across all splits

    Key features:
    - Walk-forward validation: Trains on past, tests on future, rolls forward
    - Triple-barrier execution: Each trade exits at TP, SL, or max holding period
    - Meta model filtering: Optional second-level model to filter out bad signals
    - Trend scanning: Monitor strategy health by watching Sharpe deterioration
    - Feature contributions: Log which features drove each trade decision
    - Resumable: Continues from last completed split if interrupted

    The backtest creates Position records for each trade with entry/exit details,
    computes split-level and overall metrics (Sharpe, drawdown, win rate), and
    optionally runs trend scanning to detect when the strategy stops working.

    Typical usage:
        python manage.py ml_backtest --config-code-name my_strategy \\
            --train-window-bars 5000 --test-window-bars 500 --step-bars 500 \\
            --probability-threshold 0.6 --enable-trend-scan
    """

    help = "Run walk-forward backtest on ML model."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, required=True)
        parser.add_argument("--train-window-bars", type=int, default=5000, help="Train on last N bars")
        parser.add_argument("--test-window-bars", type=int, default=500, help="Test on next N bars")
        parser.add_argument("--step-bars", type=int, default=500, help="Step forward by N bars")
        parser.add_argument("--n-estimators", type=int, default=500)
        parser.add_argument("--probability-threshold", type=float, default=0.6)
        parser.add_argument("--costs-bps", type=float, default=5)
        parser.add_argument("--use-meta", action="store_true", help="Use meta model for signal filtering")
        parser.add_argument("--meta-prob-threshold", type=float, default=0.5, help="Meta model probability threshold")
        parser.add_argument("--enable-trend-scan", action="store_true", help="Enable trend scanning (AFML Ch. 15)")
        parser.add_argument("--trend-scan-returns-type", type=str, default="predictions", help="Returns type: predictions, realized_pnl, meta_filtered")
        parser.add_argument("--log-feature-contributions", action="store_true", help="Log per-trade feature contributions using TreeInterpreter")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options["config_code_name"]
        train_window_bars = options["train_window_bars"]
        test_window_bars = options["test_window_bars"]
        step_bars = options["step_bars"]
        n_estimators = options["n_estimators"]
        prob_threshold = options["probability_threshold"]
        costs_bps = options["costs_bps"]
        use_meta = options["use_meta"]
        meta_prob_threshold = options["meta_prob_threshold"]
        enable_trend_scan = options["enable_trend_scan"]
        trend_scan_returns_type = options["trend_scan_returns_type"]
        log_feature_contributions = options["log_feature_contributions"]

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        candle = cfg.candle
        cfg_json = cfg.json_data

        timestamp_from = pd.to_datetime(cfg_json["timestamp_from"])
        timestamp_to = pd.to_datetime(cfg_json["timestamp_to"])

        pt_mult = cfg_json.get("pt_mult", 2.0)
        sl_mult = cfg_json.get("sl_mult", 1.0)
        max_holding_bars = cfg_json.get("max_holding_bars", 48)

        logger.info(f"{cfg}: starting walk-forward backtest from {timestamp_from} to {timestamp_to}")

        feature_data_list = MLFeatureData.objects.filter(
            candle=candle,
            timestamp_from__gte=timestamp_from,
            timestamp_to__lte=timestamp_to
        ).order_by("timestamp_from")

        if not feature_data_list.exists():
            logger.error(f"{cfg}: no feature data found")
            return

        dfs = []
        for fd in feature_data_list:
            df = pd.read_parquet(fd.file_data.open())
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data = data.sort_values("timestamp").reset_index(drop=True)

        if "label" not in data.columns:
            logger.error(f"{cfg}: no labels found")
            return

        drop_cols = ["timestamp", "label", "event_end_idx", "event_end_time", "sample_weight"]
        feature_cols = [c for c in data.columns if c not in drop_cols]

        incomplete_run = MLRun.objects.filter(
            ml_config=cfg, status="running"
        ).order_by("-created_at").first()

        if incomplete_run:
            ml_run = incomplete_run

            split_results = ml_run.metrics.get("splits", []) if ml_run.metrics else []
            if split_results:
                last_split = split_results[-1]
                current_idx = last_split.get("next_idx", 0)
                logger.info(f"{cfg}: resuming backtest from bar index {current_idx}")
            else:
                current_idx = 0
                partial_positions = Position.objects.filter(ml_run=ml_run)
                if partial_positions.exists():
                    count = partial_positions.count()
                    logger.warning(f"{cfg}: found {count} partial positions with no saved splits - deleting to prevent duplication")
                    partial_positions.delete()
                logger.info(f"{cfg}: resuming backtest from bar index 0 (no completed splits)")

            existing_positions = Position.objects.filter(ml_run=ml_run).order_by("entry_timestamp")

            cost = Decimal(str(costs_bps / 10000))
            equity = Decimal("1.0")
            all_trades = []
            for p in existing_positions:
                if p.side == 1:
                    pnl = (p.exit_price - p.entry_price) / p.entry_price if p.exit_price else Decimal("0")
                else:
                    pnl = (p.entry_price - p.exit_price) / p.entry_price if p.exit_price else Decimal("0")

                pnl = pnl - (cost * 2)

                equity *= (Decimal("1.0") + pnl)

                all_trades.append({
                    "entry_idx": 0,
                    "exit_idx": 0,
                    "direction": p.side,
                    "pnl": float(pnl),
                    "equity": float(equity),
                    "position_id": p.id
                })

            current_equity = equity
        else:
            ml_run = MLRun.objects.create(
                ml_config=cfg,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                metrics={},
                feature_importances={},
                status="running"
            )
            current_idx = 0
            all_trades = []
            split_results = []
            current_equity = Decimal("1.0")

        while current_idx + train_window_bars + test_window_bars <= len(data):
            train_start_idx = current_idx
            train_end_idx = train_start_idx + train_window_bars
            test_end_idx = train_end_idx + test_window_bars

            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]

            if len(train_data) < 100 or len(test_data) < 10:
                current_idx += step_bars
                continue

            train_start = train_data["timestamp"].iloc[0]
            train_end = train_data["timestamp"].iloc[-1]
            test_end = test_data["timestamp"].iloc[-1]

            X_train = train_data[feature_cols].fillna(0).values
            y_train = train_data["label"].values
            w_train = train_data["sample_weight"].values if "sample_weight" in train_data.columns else None

            X_test = test_data[feature_cols].fillna(0).values
            prices_test = test_data["close"].values
            timestamps_test = test_data["timestamp"].values
            highs_test = test_data["high"].values if "high" in test_data.columns else None
            lows_test = test_data["low"].values if "low" in test_data.columns else None

            vol_span = cfg_json.get("vol_span", 20)
            volatilities_test = compute_ewma_volatility(test_data, span=vol_span).values

            # Train uncalibrated model (always, for TreeInterpreter)
            primary_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

            primary_model.fit(X_train, y_train, sample_weight=w_train)

            # Try to load calibrated model for predictions
            calibrated_artifact = MLArtifact.objects.filter(
                ml_run__ml_config=cfg,
                artifact_type="calibrated_model"
            ).order_by("-ml_run__created_at").first()

            if calibrated_artifact and calibrated_artifact.artifact:
                try:
                    prediction_model = joblib.load(calibrated_artifact.artifact.open())
                    logger.info(
                        f"{cfg}: using calibrated model for predictions, "
                        f"uncalibrated model for interpretability"
                    )
                except Exception as e:
                    logger.warning(f"{cfg}: failed to load calibrated model: {e}")
                    prediction_model = primary_model
            else:
                prediction_model = primary_model
                logger.info(f"{cfg}: using uncalibrated model for predictions and interpretability")

            meta_model = None
            meta_feature_cols = None
            if use_meta:
                meta_artifact = MLArtifact.objects.filter(
                    ml_run__ml_config=cfg,
                    artifact_type="meta_model"
                ).order_by("-ml_run__created_at").first()

                if meta_artifact and meta_artifact.artifact:
                    try:
                        meta_model = joblib.load(meta_artifact.artifact.open())
                        meta_feature_cols = meta_artifact.ml_run.metadata.get("meta_feature_columns", [])
                        logger.info(f"{cfg}: loaded meta model from run {meta_artifact.ml_run.id}")
                    except Exception as e:
                        logger.warning(f"{cfg}: failed to load meta model: {e}")

            test_data_reset = test_data.reset_index(drop=True)
            signals_list = generate_signals(prediction_model, test_data_reset, prob_threshold, feature_cols)

            signals = [0] * len(test_data_reset)
            timestamp_to_idx = {ts: idx for idx, ts in enumerate(test_data_reset["timestamp"])}

            for sig in signals_list:
                idx = timestamp_to_idx.get(sig["timestamp"])
                if idx is not None:
                    primary_side = sig["prediction"]

                    if meta_model is not None and primary_side != 0:
                        row = test_data_reset.iloc[idx]
                        meta_features = row.copy()
                        meta_features["primary_side"] = primary_side

                        X_meta = meta_features[meta_feature_cols].fillna(0).values.reshape(1, -1)

                        try:
                            meta_pred_proba = meta_model.predict_proba(X_meta)[0]
                            meta_prob = meta_pred_proba[1] if len(meta_pred_proba) > 1 else 0.0
                            meta_label = 1 if meta_prob >= meta_prob_threshold else 0

                            if meta_label == 0:
                                logger.debug(f"{cfg}: meta model filtered signal at {sig['timestamp']}")
                                continue
                        except Exception as e:
                            logger.warning(f"{cfg}: meta model prediction failed: {e}")

                    signals[idx] = primary_side

            feature_contributions = None
            if log_feature_contributions:
                signal_indices = [i for i, s in enumerate(signals) if s != 0]

                if signal_indices:
                    X_signals = X_test[signal_indices]
                    # Use primary_model (uncalibrated RF) for TreeInterpreter
                    feature_contributions = compute_tree_contributions(primary_model, X_signals, feature_cols)

                    contrib_summary = get_feature_contribution_summary(feature_contributions, top_k=3)
                    for i, feat_info in enumerate(contrib_summary["top_features"]):
                        logger.info(
                            f"{cfg}: trade {i+1} top feature: {feat_info['feature']} "
                            f"(contrib: {feat_info['mean_abs_contribution']:.4f})"
                        )

            trades, current_equity = simulate_trades(
                signals, prices_test, timestamps_test, ml_run,
                costs_bps=costs_bps,
                pt_mult=pt_mult, sl_mult=sl_mult, max_holding_bars=max_holding_bars,
                highs=highs_test, lows=lows_test, volatilities=volatilities_test,
                starting_equity=current_equity,
                meta_used=(meta_model is not None),
                feature_contributions=feature_contributions
            )
            metrics = compute_metrics(trades, current_equity)

            all_trades.extend(trades)

            next_idx = current_idx + step_bars

            split_metadata = {
                "train_start_idx": train_start_idx,
                "train_end_idx": train_end_idx,
                "test_end_idx": test_end_idx,
                "next_idx": next_idx,
                "train_start": str(train_start),
                "train_end": str(train_end),
                "test_start": str(train_end),
                "test_end": str(test_end),
                "metrics": metrics
            }

            if enable_trend_scan and test_end_idx > 0:
                trend_scan_cfg = cfg_json.get("trend_scan", {})
                window_sizes = trend_scan_cfg.get("window_sizes", [250, 500, 1000])
                window_step = trend_scan_cfg.get("step", 100)
                min_obs = trend_scan_cfg.get("min_obs", 100)
                method = trend_scan_cfg.get("method", "sharpe")
                embargo_bars_scan = cfg_json.get("embargo_bars", 10)

                oos_data = data.iloc[:test_end_idx]

                if trend_scan_returns_type == "predictions":
                    probs = prediction_model.predict_proba(data.iloc[:test_end_idx][feature_cols].fillna(0).values)
                    pred_returns = np.array([p[1] - p[0] if len(p) > 1 else 0.0 for p in probs])
                elif trend_scan_returns_type == "realized_pnl":
                    pred_returns = np.array([t["pnl"] for t in all_trades])
                elif trend_scan_returns_type == "meta_filtered":
                    meta_trades = [t for t in all_trades if t.get("meta_filtered", True)]
                    pred_returns = np.array([t["pnl"] for t in meta_trades]) if meta_trades else np.array([])
                else:
                    pred_returns = None

                if pred_returns is not None and len(pred_returns) > min_obs:
                    scan_df = pd.DataFrame({
                        'timestamp': oos_data['timestamp'].values[:len(pred_returns)]
                    })

                    windows = generate_trend_windows(scan_df, window_sizes, window_step, min_obs)

                    weights = oos_data["sample_weight"].values[:len(pred_returns)] if "sample_weight" in oos_data.columns else None
                    event_end_idx = oos_data["event_end_idx"].values[:len(pred_returns)] if "event_end_idx" in oos_data.columns else None

                    scan_results = scan_trends(
                        pred_returns,
                        weights,
                        event_end_idx,
                        windows,
                        embargo_bars_scan,
                        method
                    )

                    top_trends = rank_trends(scan_results, top_k=5)

                    for trend_result in top_trends:
                        window = trend_result['window']
                        stat = trend_result['statistic']

                        TrendScan.objects.create(
                            ml_config=cfg,
                            ml_run=ml_run,
                            timestamp=test_end,
                            window_start_idx=window['start_idx'],
                            window_end_idx=window['end_idx'],
                            window_size=window['size'],
                            timestamp_start=window['timestamp_start'],
                            timestamp_end=window['timestamp_end'],
                            score=float(stat['score']),
                            mean_return=float(stat['mean']),
                            std_return=float(stat['std']),
                            p_value=float(stat['p_value']),
                            n_events=trend_result['n_events'],
                            method=method,
                            returns_type=trend_scan_returns_type
                        )

                    split_metadata["trend_scans"] = [
                        {
                            "window_size": t['window']['size'],
                            "score": t['statistic']['score'],
                            "n_events": t['n_events']
                        }
                        for t in top_trends
                    ]

            split_results.append(split_metadata)

            all_trades.extend(trades)

            logger.info(f"{cfg}: bars [{train_start_idx}:{test_end_idx}] ({train_end} -> {test_end}) - {metrics['total_trades']} trades, equity: {metrics['final_equity']:.4f}")

            ml_run.metrics = {
                "splits": split_results
            }
            ml_run.save(update_fields=["metrics"])

            from quant_tick.models import CandleData
            checkpoint = CandleData.objects.filter(
                candle=candle, timestamp__lte=test_end
            ).order_by("-timestamp").first()
            if checkpoint:
                cfg.last_candle_data = checkpoint
                cfg.save(update_fields=["last_candle_data"])

            current_idx += step_bars

        if all_trades:
            final_equity = Decimal(str(all_trades[-1]["equity"]))
        else:
            final_equity = Decimal("1.0")

        overall_metrics = compute_metrics(all_trades, final_equity)

        ml_run.metrics = {
            "overall": overall_metrics,
            "splits": split_results
        }
        ml_run.status = "completed"
        ml_run.save()

        logger.info(f"{cfg}: backtest complete - Final equity: {overall_metrics['final_equity']:.4f}, Sharpe: {overall_metrics['sharpe_ratio']:.2f}")
