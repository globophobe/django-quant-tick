from datetime import datetime

import pandas as pd
from django.db import transaction
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import Frequency
from quant_tick.lib import filter_by_timestamp, get_min_time, iter_window

from ..candles import Candle, CandleCache, CandleData


class TimeBasedCandle(Candle):
    """Fixed-window candles."""

    def get_trade_candle(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        trade_data,
    ) -> dict | None:
        """Return a trade candle for aligned daily-or-higher slices."""
        window = self.json_data["window"]
        window_minutes = int(pd.Timedelta(window).total_seconds() / 60)
        if window_minutes < Frequency.DAY:
            return None
        if self.json_data.get("min_volume_exponent") != 1:
            return None
        if self.json_data.get("min_notional_exponent") != 1:
            return None
        if trade_data.json_data is None or "candle" not in trade_data.json_data:
            return None
        trade_source = trade_data.get_candle_source_data()
        if trade_source != self.json_data["source_data"]:
            return None
        obj_from = trade_data.timestamp
        obj_to = trade_data.timestamp + pd.Timedelta(f"{trade_data.frequency}min")
        if timestamp_from != obj_from or timestamp_to != obj_to:
            return None
        return dict(trade_data.json_data["candle"])

    def get_max_timestamp_to(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> datetime:
        """Clamp timestamp_to to the last complete window."""
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
        trade_candle: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate complete windows and keep any remainder in cache."""
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

        for window_from, window_to in iter_window(ts_from, max_ts_to, window):
            ts_to = window_to
            candle = None
            if trade_candle is not None:
                candle = dict(trade_candle)
                candle["timestamp"] = window_from
            elif not data_frame.empty:
                df = filter_by_timestamp(data_frame, window_from, window_to)
                if len(df):
                    candle = self._aggregate_candle(df, timestamp=window_from)

            if candle is not None:
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
            if trade_candle is not None:
                values = dict(trade_candle)
                values["timestamp"] = cache_ts_from
                if "next" in cache_data:
                    previous_values = cache_data.pop("next")
                    cache_data["next"] = self._merge_cache(previous_values, values)
                else:
                    cache_data["next"] = values
            elif not data_frame.empty:
                cache_df = filter_by_timestamp(data_frame, cache_ts_from, timestamp_to)
                if len(cache_df):
                    cache_data = self._get_next_cache(
                        cache_df, cache_data, timestamp=cache_ts_from
                    )

        return data, cache_data

    @transaction.atomic
    def on_retry(self, timestamp_from: datetime, timestamp_to: datetime) -> None:
        """Delete cached and stored rows inside the retry window."""
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
