from datetime import datetime
from typing import Optional, Tuple

from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_candle, get_key, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle, CandleCache


class ConstantCandle(Candle):
    def initialize(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        step: str,
        retry: bool = False,
    ) -> Tuple[datetime, datetime, Optional[dict], Optional[DataFrame]]:
        """Initialize."""
        candle_cache = (
            CandleCache.objects.filter(candle=self, timestamp__lt=timestamp_from)
            .only("timestamp", "json_data")
            .first()
        )
        if candle_cache:
            data = candle_cache.json_data
            data_frame = candle_cache.get_data_frame()
        else:
            data, data_frame = self.get_initial_cache(timestamp_from)
        return timestamp_from, timestamp_to, data, data_frame

    def get_initial_cache(
        self, timestamp: datetime
    ) -> Tuple[Optional[dict], Optional[DataFrame]]:
        """Get initial cache."""
        data = {
            "date": timestamp.date(),
            "sample_type": self.json_data["sample_type"],
            "sample_value": 0,
        }
        return data, None

    def get_cache(
        self,
        timestamp: datetime,
        data: Optional[dict] = None,
        data_frame: Optional[DataFrame] = None,
    ) -> Tuple[Optional[dict], Optional[DataFrame]]:
        """Get cache."""
        frequency = self.json_data.get("frequency")
        date = timestamp.date()
        is_same_day = data["date"] == date
        # Reset cache for new era
        if not is_same_day:
            if frequency == Frequency.DAY:
                return self.get_initial_cache(timestamp)
            elif frequency == Frequency.WEEK and date.weekday() == 0:
                return self.get_initial_cache(timestamp)
        return data, data_frame

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
        cache_data_frame: Optional[DataFrame] = None,
    ) -> Tuple[list, Optional[dict], Optional[DataFrame]]:
        """Aggregate."""
        start = 0
        data = []
        topN = self.json_data.get("topN", 0)
        key = get_key(data_frame, self.json_data["sample_type"])
        for index, row in data_frame.iterrows():
            cache_data["sample_value"] += row[key]
            if self.should_aggregate_candle(data_frame, cache_data, cache_data_frame):
                df = data_frame.loc[start:index]
                candle = aggregate_candle(
                    df, sample_type=self.json_data["sample_type"], top_n=topN
                )
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(
                        previous, candle, self.json_data["sample_type"], topN
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
                df, cache_data, cache_data_frame, self.json_data["sample_type"], topN
            )
        return data, cache_data, cache_data_frame

    def should_aggregate_candle(
        self,
        data_frame: DataFrame,
        cache_data: dict,
        cache_data_frame: Optional[DataFrame] = None,
    ) -> bool:
        """Should aggregate candle."""
        return cache_data["sample_value"] >= self.json_data["sample_value"]

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
