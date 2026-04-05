from datetime import datetime
from decimal import Decimal

import pandas as pd
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import Frequency

from ..candles import Candle


class ConstantCandle(Candle):
    """Constant-threshold candles.

    These candles close when cumulative ticks, volume, or notional reaches
    `target_value`. Example: a volume candle with `target_value=10_000`
    closes each time 10,000 units trade.

    Optional `cache_reset` settings restart the running total on day or week
    boundaries.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        cache = {}
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        cache["sample_value"] = 0
        return cache

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Reset cache at configured day/week boundaries."""
        if self.should_reset_cache(timestamp, data):
            data = self.get_initial_cache(timestamp)
        return data

    def should_reset_cache(self, timestamp: datetime, data: dict) -> bool:
        """Whether cache_reset requires a new running total."""
        date = timestamp.date()
        cache_date = data.get("date")
        cache_reset = self.json_data.get("cache_reset")
        is_daily_reset = cache_reset == Frequency.DAY
        is_weekly_reset = cache_reset == Frequency.WEEK and date.weekday() == 0
        if cache_date:
            is_same_day = cache_date == date
            if not is_same_day and (is_daily_reset or is_weekly_reset):
                return True
        return False

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate threshold-based candles over one trade-data slice."""
        data_frame = self._preprocess_data(data_frame, cache_data)

        start = 0
        data = []
        for index, row in data_frame.iterrows():
            cache_data["sample_value"] += self.get_sample_value(row)
            if self.should_aggregate_candle(cache_data):
                df = data_frame.loc[start:index]
                candle = self._aggregate_candle(df)
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    if "open" in previous:
                        candle = self._merge_cache(previous, candle)
                data.append(candle)
                # Reinitialize cache
                cache_data["sample_value"] = 0
                # Next index
                start = index + 1

        # Cache incomplete candle
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache_data = self._get_next_cache(df, cache_data)

        data, cache_data = self.get_incomplete_candle(timestamp_to, data, cache_data)
        return data, cache_data

    def get_sample_value(self, row: tuple) -> Decimal | int:
        sample_type = self.json_data["sample_type"]
        key = "total" + sample_type.title()
        if key in row.index:
            return row[key]
        return row[sample_type]

    def should_aggregate_candle(self, data: dict) -> bool:
        return data["sample_value"] >= self.json_data["target_value"]

    def get_incomplete_candle(
        self, timestamp: datetime, data: list, cache_data: dict
    ) -> tuple[list, dict]:
        """Get incomplete candle.

        Saved only if cache resets next iteration.
        """
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
