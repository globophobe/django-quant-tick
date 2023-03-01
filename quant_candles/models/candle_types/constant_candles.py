from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle, CandleCache


class ConstantCandle(Candle):
    """Constant candle.

    For example, 1 candle every:
    * 1000 ticks.
    * 10,000 notional.
    * 1,000,000 dollars.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {}
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        cache["sample_value"] = 0
        return cache

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data."""
        if self.should_reset_cache(timestamp, data):
            data = self.get_initial_cache(timestamp)
        return data

    def should_reset_cache(self, timestamp: datetime, data: dict) -> bool:
        """Should reset cache."""
        date = timestamp.date()
        cache_date = data.get("date")
        cache_reset = self.json_data.get("cache_reset")
        is_daily_reset = cache_reset == Frequency.DAY
        is_weekly_reset = cache_reset == Frequency.WEEK and date.weekday() == 0
        if cache_date:
            is_same_day = cache_date == date
            if not is_same_day:
                if is_daily_reset or is_weekly_reset:
                    return True
        return False

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        can_agg = super().can_aggregate(timestamp_from, timestamp_to)
        last_cache = (
            CandleCache.objects.filter(candle=self)
            .only("timestamp", "frequency")
            .last()
        )
        if last_cache:
            ts_from = last_cache.timestamp + pd.Timedelta(f"{last_cache.frequency}t")
            # Don't aggregate without last cache, if not first iteration.
            has_candle_cache = (
                timestamp_from == ts_from
                or not CandleCache.objects.filter(candle=self).exists()
            )
            return can_agg and has_candle_cache
        else:
            return True

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
        data, cache_data = self.get_incomplete_candle(timestamp_to, data, cache_data)
        return data, cache_data

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= self.json_data["target_value"]

    def get_incomplete_candle(
        self, timestamp: datetime, data: list, cache_data: dict
    ) -> tuple[list, dict]:
        """Get incomplete candle."""
        ts = timestamp + pd.Timedelta("1us")
        if self.should_reset_cache(ts, cache_data):
            if "next" in cache_data:
                candle = cache_data.pop("next")
                candle["incomplete"] = True
                data.append(candle)
        return data, cache_data

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
