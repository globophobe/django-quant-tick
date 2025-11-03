"""Manual trend scanning command for analyzing existing data."""
import logging
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone

from quant_tick.lib.trend_scanning import (
    generate_trend_windows,
    rank_trends,
    scan_trends,
)
from quant_tick.models import MLConfig, Position, TrendScan

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Run trend scanning on historical position data."""

    help = "Run trend scanning on historical positions to detect structural trends."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, required=True)
        parser.add_argument("--position-type", type=str, default="live", help="Position type: live or backtest")
        parser.add_argument("--max-positions", type=int, default=1000, help="Max positions to analyze")
        parser.add_argument("--window-sizes", type=str, default="250,500,1000", help="Comma-separated window sizes")
        parser.add_argument("--step", type=int, default=100, help="Step between windows")
        parser.add_argument("--min-obs", type=int, default=100, help="Minimum observations per window")
        parser.add_argument("--method", type=str, default="sharpe", help="Method: sharpe or t_stat")
        parser.add_argument("--top-k", type=int, default=10, help="Number of top trends to save")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options["config_code_name"]
        position_type = options["position_type"]
        max_positions = options["max_positions"]
        window_sizes = [int(x) for x in options["window_sizes"].split(",")]
        step = options["step"]
        min_obs = options["min_obs"]
        method = options["method"]
        top_k = options["top_k"]

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        logger.info(f"{cfg}: running trend scan on {position_type} positions")

        positions = Position.objects.filter(
            ml_run__ml_config=cfg,
            position_type=position_type,
            exit_timestamp__isnull=False
        ).order_by("entry_timestamp")[:max_positions]

        if positions.count() < min_obs:
            logger.error(f"{cfg}: insufficient positions ({positions.count()}/{min_obs})")
            return

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

        logger.info(f"{cfg}: analyzing {len(returns)} positions")

        df = pd.DataFrame({'timestamp': timestamps})
        windows = generate_trend_windows(df, window_sizes, step, min_obs)

        logger.info(f"{cfg}: scanning {len(windows)} candidate windows")

        scan_results = scan_trends(returns, None, None, windows, 0, method)
        top_trends = rank_trends(scan_results, top_k=top_k)

        now = timezone.now()
        saved = 0
        for trend_result in top_trends:
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
            saved += 1

            logger.info(
                f"{cfg}: trend window [{window['start_idx']}:{window['end_idx']}] "
                f"size={window['size']}, score={stat['score']:.3f}, "
                f"mean={stat['mean']:.5f}, std={stat['std']:.5f}, "
                f"p_value={stat['p_value']:.3f}, n_events={trend_result['n_events']}"
            )

        logger.info(f"{cfg}: saved {saved} trend scan results")
