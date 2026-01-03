from datetime import datetime
from decimal import Decimal

import pandas as pd

from quant_tick.constants import Frequency
from quant_tick.lib import get_existing, get_min_time
from quant_tick.utils import gettext_lazy as _

from ..trades import TradeData
from .constant_candles import ConstantCandle


class AdaptiveCandle(ConstantCandle):
    """Dynamic-threshold bars that adapt to recent average activity levels.

    Instead of using a fixed threshold like constant bars, adaptive bars adjust their
    threshold based on recent market conditions. The threshold is the moving average
    of activity divided by target bars per day.

    How it works:
    - Define a lookback period (e.g., 7 days, 50 days)
    - Compute moving average of daily activity (ticks, volume, or dollar)
    - Set threshold = MA / target_candles_per_day
    - Close bar when accumulated activity >= threshold

    For example, with 7-day MA and target of 24 bars/day:
    - If recent avg is 240,000 ticks/day, threshold = 240,000 / 24 = 10,000 ticks/bar
    - If market gets busier and avg rises to 480,000 ticks/day, threshold rises to 20,000
    - This keeps bar frequency stable even as overall market activity changes

    Why use adaptive bars? They maintain consistent bar frequency across different
    market regimes. During bull markets with high activity, fixed thresholds would
    create too many bars. During bear markets with low activity, fixed thresholds would
    create too few bars. Adaptive bars solve this by tracking the moving average.

    Common configurations:
    - Short-term adaptive: 7-day MA, 24 bars/day (intraday trading)
    - Long-term adaptive: 50-day MA, 6 bars/day (swing trading)
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
            data["target_value"] = self.get_moving_average_value(timestamp)
        return data

    def get_trade_data_for_moving_average(self, timestamp: datetime) -> bool:
        """Get trade data for moving average."""
        days = self.json_data["moving_average_number_of_days"]
        delta = pd.Timedelta(f"{days}d")
        ts = get_min_time(timestamp, value="1d") - delta
        return TradeData.objects.filter(
            symbol=self.symbol, timestamp__gte=ts, timestamp__lt=ts + delta
        )

    def get_moving_average_value(self, timestamp: datetime) -> Decimal:
        """Get moving average value."""
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
        """Can aggregate."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        trade_data_ma = self.get_trade_data_for_moving_average(timestamp_from)
        existing = get_existing(trade_data_ma.values("timestamp", "frequency"))
        days = self.json_data["moving_average_number_of_days"]
        can_calculate_ma = len(existing) == Frequency.DAY * days
        return can_agg and can_calculate_ma

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= data["target_value"]

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
