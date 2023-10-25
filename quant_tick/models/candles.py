import os
from datetime import datetime
from itertools import chain
from typing import List, Optional, Tuple

import pandas as pd
from django.conf import settings
from django.db import models
from django.db.models import Q, QuerySet
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_tick.constants import Frequency
from quant_tick.lib import (
    filter_by_timestamp,
    get_current_time,
    get_existing,
    get_min_time,
    get_runs,
    has_timestamps,
    parse_datetime,
)
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, AbstractDataStorage, JSONField
from .trades import TradeData, TradeDataSummary


def upload_data_to(instance: "CandleData", filename: str) -> str:
    """Upload to."""
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    prefix = f"candles/{instance.candle.code_name}/{date}"
    return f"{prefix}/{fname}{ext}"


class Candle(AbstractCodeName, PolymorphicModel):
    symbols = models.ManyToManyField(
        "quant_tick.Symbol",
        db_table="quant_candles_candle_symbol",
        verbose_name=_("symbols"),
    )
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    def initialize(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> Tuple[datetime, datetime, dict]:
        """Initialize."""
        # Is there a specific date from?
        if self.date_from:
            min_timestamp_from = parse_datetime(self.date_from)
            ts_from = (
                min_timestamp_from
                if timestamp_from < min_timestamp_from
                else timestamp_from
            )
        else:
            ts_from = timestamp_from
        # Is there a specific date to?
        if self.date_to:
            max_timestamp_to = parse_datetime(self.date_to)
            ts_to = (
                max_timestamp_to if timestamp_to < max_timestamp_to else timestamp_to
            )
        else:
            ts_to = timestamp_to
        # Does it have trade data?
        max_ts_to = min(
            [
                t.timestamp + pd.Timedelta(f"{t.frequency}t")
                for symbol in self.symbols.all()
                if (
                    t := TradeData.objects.filter(symbol=symbol)
                    .only("timestamp", "frequency")
                    .last()
                )
            ],
            default=ts_to,
        )
        ts_to = ts_to if max_ts_to > ts_to else max_ts_to
        # Does it have a cache?
        candle_cache = (
            CandleCache.objects.filter(
                candle=self,
                timestamp__lte=ts_to,
            )
            .order_by("-timestamp")
            .only("timestamp", "frequency", "json_data")
            .first()
        )
        if candle_cache:
            if not retry:
                timestamp = candle_cache.timestamp + pd.Timedelta(
                    f"{candle_cache.frequency}t"
                )
                ts_from = timestamp if timestamp > ts_from else ts_from
            data = candle_cache.json_data
        else:
            data = self.get_initial_cache(ts_from)
        return ts_from, ts_to, data

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        return {}

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data."""
        return data

    def get_data_frame(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get data frame."""
        # Trade data may be hourly, so timestamp from >= hourly timestamp.
        trade_data = (
            TradeData.objects.filter(
                symbol__in=self.symbols.all(),
                timestamp__gte=get_min_time(timestamp_from, value="1h"),
                timestamp__lt=timestamp_to,
            )
            .select_related("symbol")
            .only("symbol", "timestamp", "frequency", "file_data")
        )
        data_frames = []
        for symbol in self.symbols.all():
            target = sorted(
                [obj for obj in trade_data if obj.symbol == symbol],
                key=lambda obj: obj.timestamp,
            )
            dfs = []
            for t in target:
                # Query may contain trade data by minute.
                # Only get target timestamps.
                if timestamp_from <= t.timestamp + pd.Timedelta(f"{t.frequency}t"):
                    df = t.get_data_frame()
                    if df is not None:
                        dfs.append(df)
            if dfs:
                df = pd.concat(dfs)
                df.insert(2, "exchange", symbol.exchange)
                df.insert(3, "symbol", symbol.symbol)
                data_frames.append(df)
        if data_frames:
            df = pd.concat(data_frames).sort_values(["timestamp", "nanoseconds"])
            return (
                filter_by_timestamp(df, timestamp_from, timestamp_to)
                .reset_index()
                .drop(columns=["index"])
            )
        else:
            return pd.DataFrame([])

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        values = []
        for symbol in self.symbols.all():
            # Trade data may be hourly, so timestamp from >= hourly timestmap.
            trade_data = TradeData.objects.filter(
                symbol=symbol,
                timestamp__gte=get_min_time(timestamp_from, value="1h"),
                timestamp__lt=timestamp_to,
            )
            # Only get target timestamps.
            existing = [
                t
                for t in get_existing(trade_data.values("timestamp", "frequency"))
                if t >= timestamp_from and t < timestamp_to
            ]
            values.append(has_timestamps(timestamp_from, timestamp_to, existing))
        return all(values)

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: Optional[dict] = None,
    ) -> Tuple[list, Optional[dict]]:
        """Aggregate."""
        raise NotImplementedError

    def get_data(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        limit: Optional[int] = None,
    ) -> list:
        """Get data."""
        query = (
            Q(candle_id=self.id)
            & Q(timestamp__gte=timestamp_from)
            & Q(timestamp__lt=timestamp_to)
        )
        candle_data = (
            CandleData.objects.filter(query)
            .only("timestamp", "json_data")
            .order_by("-timestamp")
            .values("timestamp", "json_data")
        )
        if limit:
            candle_data = candle_data[:limit]
        candle_read_only_data = (
            CandleReadOnlyData.objects.filter(query)
            .order_by("-timestamp")
            .values("timestamp", "json_data")
        )
        if limit:
            candle_read_only_data = candle_read_only_data[: limit - candle_data.count()]
        return list(chain(candle_data, candle_read_only_data))

    def get_trade_data_summary(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> QuerySet:
        """Get trade data summary."""
        return TradeDataSummary.objects.filter(
            symbol__in=self.symbols.all(),
            date__gte=timestamp_from.date(),
            date__lt=timestamp_to.date(),
        )

    def get_runs(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        runs_n: int = 0,
        limit: int = 10000,
    ) -> list:
        """Get runs."""
        symbols = self.symbols.all()
        total_symbols = len(symbols)
        data_frame = pd.DataFrame([])
        this_morning_at_midnight = get_min_time(get_current_time(), value="1t")
        for symbol in symbols:
            if timestamp_to > this_morning_at_midnight:
                trade_data = TradeData.objects.filter(
                    symbol=symbol, timestamp__gte=this_morning_at_midnight
                )
                for t in trade_data.only("file_data"):
                    df = t.get_data_frame()
                    if df is not None and len(df):
                        runs = get_runs(
                            df,
                            bins=[
                                int(b * symbol.currency_divisor)
                                for b in (1_000, 10_000, 50_000, 100_000)
                            ],
                        )
                        runs_df = pd.DataFrame(runs)
                        if len(symbols) > 1:
                            runs_df.insert(2, "exchange", symbol.exchange)
                            runs_df.insert(3, "symbol", symbol.symbol)
                        data_frame = pd.concat([data_frame, runs_df])
        if len(data_frame) < limit:
            one_day = pd.Timedelta("1d")
            date = timestamp_to.date()
            while date >= timestamp_from.date():
                for symbol in symbols:
                    trade_data_summary = (
                        TradeDataSummary.objects.only("file_data")
                        .filter(symbol=symbol, date=date)
                        .distinct()
                    )
                    for t in trade_data_summary:
                        df = t.get_data_frame()
                        if df is not None and len(df):
                            if total_symbols > 1:
                                df.insert(2, "exchange", symbol.exchange)
                                df.insert(3, "symbol", symbol.symbol)
                            data_frame = pd.concat([data_frame, df])
                if len(data_frame) < limit:
                    date -= one_day
                else:
                    break
        df = data_frame.reset_index(inplace=True).sort_values(["timestamp"])
        return filter_by_timestamp(df, timestamp_from, timestamp_to, inclusive=True)

    def write_cache(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data: Optional[dict] = None,
        data_frame: Optional[DataFrame] = None,
    ):
        """Write cache.

        Delete previously saved data.
        """
        queryset = CandleCache.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_from
        )
        queryset.delete()
        delta = timestamp_to - timestamp_from
        # FIXME: Can't define every value in constants.
        frequency = delta.total_seconds() / 60
        CandleCache.objects.create(
            candle=self, timestamp=timestamp_from, frequency=frequency, json_data=data
        )

    def write_data(
        self, timestamp_from: datetime, timestamp_to: datetime, json_data: List[dict]
    ) -> None:
        """Write data.

        Delete previously saved data.
        """
        query = Q(timestamp__gte=timestamp_from) & Q(timestamp__lt=timestamp_to)
        if settings.IS_LOCAL:
            CandleReadOnlyData.objects.filter(Q(candle_id=self.id), query).delete()
        candle_data = CandleData.objects.filter(Q(candle=self), query)
        had_candle_data = candle_data.exists()
        candle_data.delete()
        data = []
        for j in json_data:
            timestamp = j.pop("timestamp")
            kwargs = {"timestamp": timestamp, "json_data": j}
            if settings.IS_LOCAL and not had_candle_data:
                c = CandleReadOnlyData(candle_id=self.id, **kwargs)
            else:
                c = CandleData(candle=self, **kwargs)
            data.append(c)
        # If local, and there was no previous cloud data, save to SQLite.
        if settings.IS_LOCAL and not had_candle_data:
            CandleReadOnlyData.objects.bulk_create(data)
        else:
            CandleData.objects.bulk_create(data)

    class Meta:
        db_table = "quant_candles_candle"
        verbose_name = _("candle")
        verbose_name_plural = _("candles")


class CandleCache(AbstractDataStorage):
    candle = models.ForeignKey(
        "quant_tick.Candle", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(
        _("frequency"), choices=Frequency.choices, db_index=True
    )
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_candles_candle_cache"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle cache")


class BaseCandleData(models.Model):
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        abstract = True


class CandleData(BaseCandleData):
    candle = models.ForeignKey(
        "quant_tick.Candle", on_delete=models.CASCADE, verbose_name=_("candle")
    )

    class Meta:
        db_table = "quant_candles_candle_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")


class CandleReadOnlyData(BaseCandleData):
    # SQLite, so not PK
    candle_id = models.BigIntegerField(verbose_name=_("candle"), db_index=True)

    class Meta:
        db_table = "quant_candles_candle_read_only_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")
