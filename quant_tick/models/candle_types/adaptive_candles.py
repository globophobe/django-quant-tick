from datetime import datetime
from decimal import Decimal

import pandas as pd
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _

from quant_tick.constants import Frequency
from quant_tick.lib import get_existing, get_min_time

from ..trades import TradeData
from .constant_candles import ConstantCandle


class AdaptiveCandle(ConstantCandle):
    """Moving-average threshold candles.

    These candles recompute `target_value` from recent daily activity and
    `target_candles_per_day`, so the threshold adapts with market activity.
    Example: if a 7-day average volume is 240,000 and the target is 24
    candles per day, each candle closes at 10,000 volume.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        return {"date": timestamp.date(), "target_value": None, "sample_value": 0}

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Refresh cache target for the current day."""
        data = super().get_cache_data(timestamp, data)
        date = timestamp.date()
        is_same_day = data["date"] == date
        has_target_value = data["target_value"] is not None
        if not is_same_day or not has_target_value:
            data["date"] = timestamp.date()
            data["target_value"] = self.get_moving_average_value(timestamp)
        return data

    def get_trade_data_for_moving_average(self, timestamp: datetime) -> QuerySet:
        """Get daily trade data for the moving-average window."""
        days = self.json_data["moving_average_number_of_days"]
        delta = pd.Timedelta(f"{days}d")
        ts = get_min_time(timestamp, value="1d") - delta
        return TradeData.objects.filter(
            symbol=self.symbol, timestamp__gte=ts, timestamp__lt=ts + delta
        )

    def get_moving_average_value(self, timestamp: datetime) -> Decimal:
        """Get the current adaptive threshold."""
        days = self.json_data["moving_average_number_of_days"]
        trade_data = (
            self.get_trade_data_for_moving_average(timestamp)
            .only("json_data")
            .values("json_data")
            .order_by("-timestamp")
        )
        sample_type = self.json_data["sample_type"]
        total = sum(
            [
                t["json_data"]["candle"][sample_type]
                for t in trade_data
                if t["json_data"] is not None
            ]
        )
        return total / days / self.json_data["target_candles_per_day"]

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Require enough daily history to compute the adaptive threshold."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        if can_agg:
            trade_data = self.get_trade_data_for_moving_average(timestamp_from)
            existing = get_existing(trade_data.values("timestamp", "frequency"))
            days = self.json_data["moving_average_number_of_days"]
            return len(existing) >= Frequency.DAY * days
        return False

    def should_aggregate_candle(self, data: dict) -> bool:
        return data["sample_value"] >= data["target_value"]

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
