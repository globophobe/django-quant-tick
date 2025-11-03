import logging
from decimal import Decimal

import numpy as np
import pandas as pd
from django.utils import timezone
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.execution.order_submission import create_and_submit_positions
from quant_tick.lib.ml import trigger_ml_inference
from quant_tick.lib.trend_scanning import (
    detect_break,
    generate_trend_windows,
    rank_trends,
    scan_trends,
)
from quant_tick.models import CandleData, MLConfig, Position, TrendAlert, TrendScan

logger = logging.getLogger(__name__)


class InferenceView(ListAPIView):
    """Inference view."""

    queryset = MLConfig.objects.filter(is_active=True)

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Run inference for each active MLConfig."""
        for cfg in self.get_queryset():
            candle = cfg.candle
            lookback_bars = cfg.json_data.get("lookback_bars", 50)

            latest_candle = (
                CandleData.objects.filter(candle=candle)
                .order_by("-timestamp")
                .first()
            )
            if not latest_candle:
                logger.info(f"{cfg}: no CandleData available, skipping")
                continue

            timestamp_to = latest_candle.timestamp

            if cfg.last_candle_data:
                timestamp_from = cfg.last_candle_data.timestamp
            else:
                total_bars = CandleData.objects.filter(candle=candle).count()
                if total_bars < lookback_bars:
                    logger.info(
                        f"{cfg}: insufficient data ({total_bars}/{lookback_bars} bars)"
                    )
                    continue

                timestamp_from = latest_candle.timestamp

            try:
                signals = trigger_ml_inference(candle, timestamp_from, timestamp_to)
                if signals:
                    count = len(signals)
                    logger.info(f"{cfg}: created {count} signals")

                    submitted = create_and_submit_positions(candle, signals)
                    if submitted > 0:
                        logger.info(f"{cfg}: submitted {submitted} positions")

                    cfg.last_candle_data = latest_candle
                    cfg.save(update_fields=["last_candle_data"])
            except Exception as e:
                logger.error(f"{cfg}: inference failed - {e}")

            if cfg.json_data.get("trend_scan", {}).get("enable_live_monitoring", False):
                try:
                    _run_trend_monitoring(cfg)
                except Exception as e:
                    logger.error(f"{cfg}: trend monitoring failed - {e}")

        return Response({"ok": True})


def _run_trend_monitoring(cfg: MLConfig) -> None:
    """Run trend scanning on recent realized returns and alert on structural breaks."""
    trend_cfg = cfg.json_data.get("trend_scan", {})
    monitoring_window = trend_cfg.get("monitoring_window_bars", 1000)
    window_sizes = trend_cfg.get("window_sizes", [250, 500, 1000])
    window_step = trend_cfg.get("step", 100)
    min_obs = trend_cfg.get("min_obs", 100)
    method = trend_cfg.get("method", "sharpe")
    threshold = trend_cfg.get("break_threshold", 1.5)
    alert_action = trend_cfg.get("alert_action", "notification")

    recent_positions = Position.objects.filter(
        ml_run__ml_config=cfg,
        position_type="live",
        exit_timestamp__isnull=False
    ).order_by("-exit_timestamp")[:monitoring_window]

    if recent_positions.count() < min_obs:
        logger.debug(f"{cfg}: insufficient positions for trend monitoring ({recent_positions.count()}/{min_obs})")
        return

    positions = list(recent_positions.reverse())

    returns = []
    timestamps = []
    for pos in positions:
        if pos.side == 1:
            pnl = (pos.exit_price - pos.entry_price) / pos.entry_price if pos.exit_price else Decimal("0")
        else:
            pnl = (pos.entry_price - pos.exit_price) / pos.entry_price if pos.exit_price else Decimal("0")
        returns.append(float(pnl))
        timestamps.append(pos.entry_timestamp)

    returns = np.array(returns)

    df = pd.DataFrame({'timestamp': timestamps})
    windows = generate_trend_windows(df, window_sizes, window_step, min_obs)

    if not windows:
        logger.debug(f"{cfg}: no valid windows for trend monitoring")
        return

    scan_results = scan_trends(returns, None, None, windows, 0, method)
    top_trends = rank_trends(scan_results, top_k=5)

    now = timezone.now()
    for trend_result in top_trends[:1]:
        window = trend_result['window']
        stat = trend_result['statistic']

        TrendScan.objects.create(
            ml_config=cfg,
            ml_run=None,
            timestamp=now,
            window_start_idx=window['start_idx'],
            window_end_idx=window['end_idx'],
            window_size=window['size'],
            timestamp_start=window['timestamp_start'],
            timestamp_end=window['timestamp_end'],
            score=Decimal(str(stat['score'])),
            mean_return=Decimal(str(stat['mean'])),
            std_return=Decimal(str(stat['std'])),
            p_value=Decimal(str(stat['p_value'])),
            n_events=trend_result['n_events'],
            method=method,
            returns_type="realized_pnl"
        )

    previous_scans = TrendScan.objects.filter(
        ml_config=cfg,
        ml_run__isnull=True,
        returns_type="realized_pnl"
    ).order_by("-timestamp")[:2]

    if previous_scans.count() >= 2:
        current_scan_records = [previous_scans[0]]
        previous_scan_records = [previous_scans[1]]

        current_scan = [
            {
                'window': {'size': s.window_size},
                'statistic': {'score': float(s.score)},
                'n_events': s.n_events
            }
            for s in current_scan_records
        ]
        previous_scan = [
            {
                'window': {'size': s.window_size},
                'statistic': {'score': float(s.score)},
                'n_events': s.n_events
            }
            for s in previous_scan_records
        ]

        break_info = detect_break(current_scan, previous_scan, threshold, top_k=5)

        if break_info['break_detected']:
            logger.warning(f"{cfg}: structural break detected - {break_info['message']}")

            TrendAlert.objects.create(
                ml_config=cfg,
                timestamp=now,
                current_top_score=Decimal(str(break_info['current_top_score'])),
                previous_top_score=Decimal(str(break_info['previous_top_score'])) if break_info['previous_top_score'] else None,
                deterioration=Decimal(str(break_info['deterioration'])) if break_info['deterioration'] else None,
                threshold=Decimal(str(threshold)),
                action=alert_action,
                window_metadata={'top_trends': [
                    {
                        'window_size': t['window']['size'],
                        'score': t['statistic']['score'],
                        'n_events': t['n_events']
                    }
                    for t in top_trends
                ]},
                status="active",
                message=break_info['message']
            )

            if alert_action == "pause_trading":
                cfg.is_active = False
                cfg.save(update_fields=["is_active"])
                logger.warning(f"{cfg}: trading paused due to structural break")
