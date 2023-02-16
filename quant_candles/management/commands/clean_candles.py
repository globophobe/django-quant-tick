import logging
from datetime import datetime

import pandas as pd
from django.contrib.contenttypes.models import ContentType
from django.db.models import Q, QuerySet

from quant_candles.constants import Frequency
from quant_candles.controllers import aggregate_candles
from quant_candles.lib import get_current_time, get_min_time, iter_timeframe
from quant_candles.management.base import BaseCandleCommand
from quant_candles.models import Candle, CandleData, CandleReadOnlyData, TimeBasedCandle
from quant_candles.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


class Command(BaseCandleCommand):
    help = "Clean candles. Candles can be cleaned if time based, or have cache reset."

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        time_based = ContentType.objects.get_for_model(
            TimeBasedCandle, for_concrete_model=False
        )
        return (
            Candle.objects.filter(
                Q(polymorphic_ctype=time_based)
                | Q(json_data__cache_reset__isnull=False)
            )
            .filter(is_active=True)
            .prefetch_related("symbols")
        )

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            candle = k["candle"]
            logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
            timestamp_from = k["timestamp_from"]
            timestamp_to = k["timestamp_to"]
            candle_data = (
                CandleData.objects.filter(candle=candle).only("timestamp").first()
            )
            candle_read_only_data = (
                CandleReadOnlyData.objects.filter(candle_id=candle.pk)
                .only("timestamp")
                .first()
            )
            candle_timestamp = min(
                [c.timestamp for c in (candle_data, candle_read_only_data) if c],
                default=None,
            )
            ts_from = max([t for t in (candle_timestamp, timestamp_from) if t])
            daily_timestamp_from = get_min_time(ts_from, value="1d")
            daily_timestamp_to = get_min_time(timestamp_to, value="1d")
            cache_reset = candle.json_data.get("cache_reset")
            if (
                isinstance(candle, TimeBasedCandle)
                or cache_reset == Frequency.DAY.value
            ):
                iterator = iter_timeframe(
                    daily_timestamp_from, daily_timestamp_to, value="1d"
                )
            elif cache_reset == Frequency.WEEK.value:
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
                    self.clean(candle, daily_ts_from, daily_ts_to)

    def clean(
        self, candle: Candle, timestamp_from: datetime, timestamp_to: datetime
    ) -> None:
        """Clean."""
        trade_data_summary = candle.get_trade_data_summary(timestamp_from, timestamp_to)
        trade_data_summary = list(trade_data_summary)
        delta = timestamp_from - timestamp_to
        if len(trade_data_summary) == delta.days:
            candles = candle.get_data(timestamp_from, timestamp_to)
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
                    [t.json_data["candle"]["high"] for t in trade_data_summary]
                ),
                "low": min([t.json_data["candle"]["low"] for t in trade_data_summary]),
            }
            for key in keys:
                expected[key] = sum(
                    [t.json_data["candle"][key] for t in trade_data_summary]
                )
            actual = {
                "high": max([c["json_data"]["high"] for c in candles]),
                "low": min([c["json_data"]["low"] for c in candles]),
            }
            for key in keys:
                actual[key] = sum([c["json_data"][key] for c in candles])

            if expected == actual:
                logging.info(
                    _("Checked {date_from} {date_to}").format(
                        **{
                            "date_from": timestamp_from.date(),
                            "date_to": timestamp_to.date(),
                        }
                    )
                )
            else:
                logger.info("{candle}: retrying...".format(**{"candle": str(candle)}))
                aggregate_candles(candle, timestamp_from, timestamp_to, retry=True)