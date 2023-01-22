from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from django.conf import settings
from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle, CandleCache, CandleData, CandleReadOnlyData


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
            is_daily_reset = reset == Frequency.DAY.value
            is_weekly_reset = reset == Frequency.WEEK.value and date.weekday() == 0
            if is_daily_reset or is_weekly_reset:
                if "next" in data:
                    ts = timestamp - pd.Timedelta("1us")
                    if settings.IS_LOCAL:
                        candle_data = CandleReadOnlyData(
                            candle_id=self.id, timestamp=ts
                        )
                    else:
                        candle_data = CandleData(candle=self, timestamp=ts)
                    d = data["next"]
                    d["incomplete"] = True
                    candle_data.json_data = d
                    candle_data.save()
                if is_daily_reset:
                    data = self.get_initial_cache(timestamp)
                elif is_weekly_reset:
                    data = self.get_initial_cache(timestamp)
        return data

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        last_candle_cache = CandleCache.objects.filter(
            candle=self,
            timestamp__gte=timestamp_from - pd.Timedelta("1h"),
            timestamp__lt=timestamp_from,
        )
        # Don't aggregate without last cache, if not first iteration.
        has_candle_cache = (
            last_candle_cache.exists()
            or not CandleCache.objects.filter(candle=self).exists()
        )
        return can_agg and has_candle_cache

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
        runs_n = self.json_data.get("runsN", None)
        top_n = self.json_data.get("topN", None)
        column = "total" + self.json_data["sample_type"].title()
        for index, row in data_frame.iterrows():
            cache_data["sample_value"] += row[column]
            if self.should_aggregate_candle(cache_data):
                df = data_frame.loc[start:index]
                candle = aggregate_candle(
                    df,
                    sample_type=self.json_data["sample_type"],
                    runs_n=runs_n,
                    top_n=top_n,
                )
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(
                        previous, candle, self.json_data["sample_type"], runs_n, top_n
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
                df, cache_data, self.json_data["sample_type"], runs_n, top_n
            )
        return data, cache_data

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= self.json_data["target_value"]

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
