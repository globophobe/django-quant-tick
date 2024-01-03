from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_tick.constants import Frequency
from quant_tick.lib import get_existing, get_min_time
from quant_tick.utils import gettext_lazy as _

from ..trades import TradeData
from .constant_candles import ConstantCandle


class AdaptiveCandle(ConstantCandle):
    """Constant candle.

    For example, 1 candle when:
    * Ticks exceed the 7 day moving average, with a target of 24 candles a day.
    * Volume exceeds the 50 day moving average, with a target of 6 candles a day.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        return {"date": timestamp.date(), "target_value": None, "sample_value": 0}

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data frame."""
        data = super().get_cache_data(timestamp, data)
        date = timestamp.date()
        is_same_day = data["date"] == date
        has_target_value = data["target_value"] is not None
        if not is_same_day or not has_target_value:
            data["date"] = timestamp.date()
            data["target_value"] = self.get_target_value(timestamp)
        return data

    def get_trade_data_for_target(self, timestamp: datetime) -> bool:
        """Get trade data for target."""
        days = self.json_data["moving_average_number_of_days"]
        delta = pd.Timedelta(f"{days}d")
        ts = get_min_time(timestamp, value="1d") - delta
        return TradeData.objects.filter(
            symbol__in=self.symbols.all(), timestamp__gte=ts, timestamp__lt=ts + delta
        )

    def get_target_value(self, timestamp: datetime) -> Decimal:
        """Get target value."""
        days = self.json_data["moving_average_number_of_days"]
        trade_data = (
            self.get_trade_data_for_target(timestamp)
            .only("json_data")
            .values("json_data")
            .order_by("-timestamp")
        )
        total_symbols = self.symbols.all().count()
        sample_type = self.json_data["sample_type"]
        total = sum(
            [
                t["json_data"]["candle"][sample_type]
                for t in trade_data
                if t["json_data"] is not None
            ]
        )
        return total / total_symbols / days / self.json_data["target_candles_per_day"]

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        trade_data_for_target = self.get_trade_data_for_target(timestamp_from)
        existing = get_existing(trade_data_for_target.values("timestamp", "frequency"))
        days = self.json_data["moving_average_number_of_days"]
        has_trade_data_for_target = len(existing) == Frequency.MINUTE * days
        return can_agg and has_trade_data_for_target

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= data["target_value"]

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
