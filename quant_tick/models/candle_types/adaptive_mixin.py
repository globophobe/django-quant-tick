from datetime import datetime

import pandas as pd
from django.db.models import QuerySet

from quant_tick.constants import Frequency
from quant_tick.lib import get_existing, get_min_time

from ..trades import TradeData


class AdaptiveMixin:
    """Adaptive mixin."""

    def get_trade_data_for_moving_average(self, timestamp: datetime) -> QuerySet:
        """Get trade data for moving average."""
        days = self.json_data["moving_average_number_of_days"]
        delta = pd.Timedelta(f"{days}d")
        ts = get_min_time(timestamp, value="1d") - delta
        return TradeData.objects.filter(
            symbol=self.symbol, timestamp__gte=ts, timestamp__lt=ts + delta
        )

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        if can_agg:
            trade_data = self.get_trade_data_for_moving_average(timestamp_from)
            existing = get_existing(trade_data.values("timestamp", "frequency"))
            days = self.json_data["moving_average_number_of_days"]
            return len(existing) >= Frequency.DAY * days
        return False
