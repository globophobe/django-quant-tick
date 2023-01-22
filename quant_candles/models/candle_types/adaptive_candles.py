from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_candles.lib import get_min_time
from quant_candles.utils import gettext_lazy as _

from ..trades import TradeDataSummary
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
        has_target_value = data.get("target_value") is not None
        if not is_same_day or not has_target_value:
            data["target_value"] = self.get_target_value(timestamp)
        return data

    def get_trade_data_summary(self, timestamp: datetime) -> bool:
        """Get trade data summary."""
        days = self.json_data["moving_average_number_of_days"]
        ts = get_min_time(timestamp, value="1d") - pd.Timedelta(f"{days}d")
        return TradeDataSummary.objects.filter(
            date__gte=ts.date(), date__lt=timestamp.date()
        )

    def get_target_value(self, timestamp: datetime) -> Decimal:
        """Get target value."""
        days = self.json_data["moving_average_number_of_days"]
        trade_data_summary = (
            self.get_trade_data_summary(timestamp)
            .only("json_data")
            .values("json_data")
            .order_by("-date")
        )
        sample_type = self.json_data["sample_type"]
        total = sum([t["json_data"]["candle"][sample_type] for t in trade_data_summary])
        return total / days / self.json_data["target_candles_per_day"]

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        trade_data_summary = self.get_trade_data_summary(timestamp_from)
        return can_agg and trade_data_summary.exists()

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= data["target_value"]

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
