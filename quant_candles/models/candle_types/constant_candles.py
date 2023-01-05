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
            "thresh_attr": self.json_data["thresh_attr"],
            "thresh_value": 0,
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
        thresh_attr = get_key(data_frame, self.json_data["thresh_attr"])
        thresh_value = self.json_data["thresh_value"]
        top_n = self.json_data.get("top_n", 0)
        for index, row in data_frame.iterrows():
            cache_data["thresh_value"] += row[thresh_attr]
            if cache_data["thresh_value"] >= thresh_value:
                df = data_frame.loc[start:index]
                candle = aggregate_candle(df, top_n)
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(previous, candle, top_n=top_n)
                data.append(candle)
                # Reinitialize cache
                cache_data["thresh_value"] = 0
                # Next index
                start = index + 1
        # Cache
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            cache_data = get_next_cache(df, cache_data, top_n)
        return data, cache_data, cache_data_frame

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
