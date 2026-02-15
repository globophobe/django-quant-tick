from datetime import datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import Frequency
from quant_tick.lib import filter_by_timestamp, get_min_time, iter_window
from quant_tick.utils import gettext_lazy as _

from ..candles import Candle, CandleCache, CandleData


class TimeBasedCandle(Candle):
    """Time based candle."""

    def get_max_timestamp_to(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> datetime:
        """Get max timestamp to."""
        window = self.json_data["window"]
        total_minutes = pd.Timedelta(window).total_seconds() / Frequency.HOUR
        delta = pd.Timedelta(f"{total_minutes}min")
        ts_to = get_min_time(timestamp_from, value="1h")
        while ts_to + delta <= timestamp_to:
            ts_to += delta
        return ts_to

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate.

        Iterate forward. Intermediate results will be saved to CandleCache.json_data
        """
        cache_data = cache_data or {}
        data = []
        window = self.json_data["window"]
        data_frame = self._preprocess_data(data_frame, cache_data)
        if "next" in cache_data:
            ts_from = cache_data["next"]["timestamp"]
        elif "next_timestamp" in cache_data:
            ts_from = cache_data.pop("next_timestamp")
        else:
            ts_from = timestamp_from

        max_ts_to = self.get_max_timestamp_to(ts_from, timestamp_to)
        ts_to = None

        for win_from, win_to in iter_window(ts_from, max_ts_to, window):
            ts_to = win_to
            if not data_frame.empty:
                df = filter_by_timestamp(data_frame, win_from, win_to)
            else:
                df = pd.DataFrame()

            if len(df):
                candle = self._build_candle(df, timestamp=win_from)

                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    if "open" in previous:
                        candle = self._merge_cache(previous, candle)
                data.append(candle)
            elif "next" in cache_data:
                candle = cache_data.pop("next")
                data.append(candle)
        # Handle incomplete window
        could_not_iterate = ts_to is None
        could_not_complete = ts_to and ts_to != timestamp_to

        if could_not_iterate or could_not_complete:
            cache_ts_from = ts_from if could_not_iterate else ts_to
            if not data_frame.empty:
                cache_df = filter_by_timestamp(data_frame, cache_ts_from, timestamp_to)
            else:
                cache_df = pd.DataFrame()

            if len(cache_df):
                cache_data = self._get_next_cache(
                    cache_df, cache_data, timestamp=cache_ts_from
                )

        return data, cache_data

    def on_retry(self, timestamp_from: datetime, timestamp_to: datetime) -> None:
        """On retry."""
        CandleCache.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        ).delete()
        CandleData.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        ).delete()

    class Meta:
        proxy = True
        verbose_name = _("time based candle")
        verbose_name_plural = _("time based candles")
