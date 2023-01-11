from datetime import datetime

import pandas as pd

from quant_candles.utils import gettext_lazy as _

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
        # Reset .
        if not is_same_day or not has_target_value:
            days = self.json_data["moving_average_number_of_days"]
            delta = pd.Timedelta(f"{days}d")
            candles = self.daily_candle.get_data(timestamp - delta, timestamp)
            sample_type = self.json_data["sample_type"]
            total = sum([c["json_data"][sample_type] for c in candles])
            data["target_value"] = (
                total / days / self.json_data["target_candles_per_day"]
            )
        return data

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= data["target_value"]

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
