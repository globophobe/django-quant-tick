import logging
from datetime import datetime

import pandas as pd
from django.contrib.contenttypes.models import ContentType
from django.db.models import Q, QuerySet

from quant_tick.constants import Frequency
from quant_tick.controllers import aggregate_candles
from quant_tick.lib import (
    get_current_time,
    get_min_time,
    is_decimal_close,
    iter_timeframe,
)
from quant_tick.management.base import BaseCandleCommand
from quant_tick.models import Candle, CandleData, TimeBasedCandle
from quant_tick.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


class Command(BaseCandleCommand):
    """Check candles."""

    help = "Candles can be checked if time based, or adaptive with cache reset."

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        time_based = ContentType.objects.get_for_model(
            TimeBasedCandle, for_concrete_model=False
        )
        return Candle.objects.filter(
            Q(polymorphic_ctype=time_based) | Q(json_data__cache_reset__isnull=False)
        ).prefetch_related("symbols")

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            candle = k["candle"]
            timestamp_from = k["timestamp_from"]
            timestamp_to = k["timestamp_to"]
            candle_data = (
                CandleData.objects.filter(candle=candle).only("timestamp").first()
            )
            candle_timestamp = candle_data.timestamp if candle_data else None
            ts_from = max([t for t in (candle_timestamp, timestamp_from) if t])
            daily_timestamp_from = get_min_time(ts_from, value="1d")
            daily_timestamp_to = get_min_time(timestamp_to, value="1d")
            cache_reset = candle.json_data.get("cache_reset")
            if isinstance(candle, TimeBasedCandle) or cache_reset == Frequency.DAY:
                iterator = iter_timeframe(
                    daily_timestamp_from, daily_timestamp_to, value="1d"
                )
            elif cache_reset == Frequency.WEEK:
                date_from = ts_from.date()
                days = 7 - date_from.weekday()
                daily_timestamp_from += pd.Timedelta(f"{days}d")
                iterator = iter_timeframe(
                    daily_timestamp_from, daily_timestamp_to, value="7d"
                )
            else:
                raise NotImplementedError
            for daily_ts_from, daily_ts_to in iterator:
                if daily_ts_to <= get_min_time(get_current_time(), value="1d"):
                    self.check_daily_range(candle, daily_ts_from, daily_ts_to)

    def check_daily_range(
        self, candle: Candle, timestamp_from: datetime, timestamp_to: datetime
    ) -> None:
        """Check daily range."""
        trade_data_summary = candle.get_expected_daily_candle(timestamp_from, timestamp_to)
        delta = timestamp_to - timestamp_from
        if len(trade_data_summary) == delta.days:
            candles = candle.get_data(timestamp_from, timestamp_to)
            if len(candles):
                candles.reverse()
                keys = (
                    "volume",
                    "buyVolume",
                    "notional",
                    "buyNotional",
                    "ticks",
                    "buyTicks",
                )
                expected = {
                    "high": max(
                        [
                            t.json_data["candle"]["high"]
                            for t in trade_data_summary
                            if t.json_data
                        ],
                        default=None,
                    ),
                    "low": min(
                        [
                            t.json_data["candle"]["low"]
                            for t in trade_data_summary
                            if t.json_data
                        ],
                        default=None,
                    ),
                }
                for key in keys:
                    expected[key] = sum(
                        [
                            t.json_data["candle"][key]
                            for t in trade_data_summary
                            if t.json_data
                        ]
                    )
                actual = {
                    "high": max(
                        [c["json_data"]["high"] for c in candles], default=None
                    ),
                    "low": min([c["json_data"]["low"] for c in candles], default=None),
                }
                for key in keys:
                    actual[key] = sum([c["json_data"][key] for c in candles])

                is_close = all(
                    [
                        (value is None and actual[key] is None)
                        or (is_decimal_close(value, actual[key]))
                        for key, value in expected.items()
                    ]
                )
                if is_close:
                    logging.info(
                        _("Checked {date_from} {date_to}").format(
                            **{
                                "date_from": timestamp_from.date(),
                                "date_to": timestamp_to.date(),
                            }
                        )
                    )
                else:
                    logger.info(
                        "{candle}: retrying...".format(**{"candle": str(candle)})
                    )
                    aggregate_candles(candle, timestamp_from, timestamp_to, retry=True)
