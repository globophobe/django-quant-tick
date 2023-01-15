from datetime import datetime
from typing import Optional, Tuple

from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class ConstantCandle(Candle):
    """Constant candle.

    For example, 1 candle every:
    * 1000 ticks.
    * 10,000 notional.
    * 1,000,000 dollars.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        return {"date": timestamp.date(), "sample_value": 0}

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data."""
        reset = self.json_data.get("cache_reset")
        date = timestamp.date()
        is_same_day = data["date"] == date
        # Reset cache.
        if not is_same_day:
            if reset == Frequency.DAY.value:
                data = self.get_initial_cache(timestamp)
            elif reset == Frequency.WEEK.value and date.weekday() == 0:
                data = self.get_initial_cache(timestamp)
        return data

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> Tuple[list, Optional[dict]]:
        """Aggregate."""
        start = 0
        data = []
        top_n = self.json_data.get("topN", 0)
        column = "total" + self.json_data["sample_type"].title()
        for index, row in data_frame.iterrows():
            cache_data["sample_value"] += row[column]
            if self.should_aggregate_candle(cache_data):
                df = data_frame.loc[start:index]
                candle = aggregate_candle(
                    df, sample_type=self.json_data["sample_type"], top_n=top_n
                )
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(
                        previous, candle, self.json_data["sample_type"], top_n
                    )
                data.append(candle)
                # Reinitialize cache
                cache_data["sample_value"] = 0
                # Next index
                start = index + 1
        # Cache
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache_data = get_next_cache(
                df, cache_data, self.json_data["sample_type"], top_n
            )
        return data, cache_data

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= self.json_data["sample_value"]

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
