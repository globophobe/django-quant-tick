from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame

from quant_candles.lib import (
    aggregate_candle,
    filter_by_timestamp,
    get_next_cache,
    iter_window,
    merge_cache,
)
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class TimeBasedCandle(Candle):
    def get_max_timestamp_to(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> datetime:
        """Get max timestamp to."""
        window = self.json_data["window"]
        total_minutes = pd.Timedelta(window).total_seconds() / 60
        delta = pd.Timedelta(f"{total_minutes}t")
        ts_to = timestamp_from
        while ts_to + delta <= timestamp_to:
            ts_to += delta
        return ts_to

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict = {},
        cache_data_frame: Optional[DataFrame] = None,
    ) -> Tuple[list, Optional[dict]]:
        """Aggregate.

        Iterate forward. Intermediate results will be saved to CandleCache.json_data
        """
        data = []
        sample_type = self.json_data.get("sample_type", None)
        runs_n = self.json_data.get("runsN", None)
        top_n = self.json_data.get("topN", None)
        window = self.json_data["window"]
        if "next" in cache_data:
            ts_from = cache_data["next"]["timestamp"]
        else:
            ts_from = timestamp_from
        max_ts_to = self.get_max_timestamp_to(ts_from, timestamp_to)
        ts_to = None
        for ts_from, ts_to in iter_window(ts_from, max_ts_to, window):
            df = filter_by_timestamp(data_frame, ts_from, ts_to)
            if len(df):
                candle = aggregate_candle(
                    data_frame, timestamp=ts_from, runs_n=runs_n, top_n=top_n
                )
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(previous, candle, sample_type, runs_n, top_n)
                data.append(candle)
            elif "next" in cache_data:
                candle = cache_data.pop("next")
                data.append(candle)
        could_not_iterate = ts_to is None
        could_not_complete_iteration = ts_to and ts_to != timestamp_to
        if could_not_iterate or could_not_complete_iteration:
            if could_not_iterate:
                cache_ts_from = ts_from
            else:
                cache_ts_from = ts_to
            cache_df = filter_by_timestamp(data_frame, cache_ts_from, timestamp_to)
            if len(cache_df):
                cache_data = get_next_cache(
                    cache_df,
                    cache_data,
                    timestamp=cache_ts_from,
                    sample_type=sample_type,
                    runs_n=runs_n,
                    top_n=top_n,
                )
        return data, cache_data

    class Meta:
        proxy = True
        verbose_name = _("time based candle")
        verbose_name_plural = _("time based candles")
