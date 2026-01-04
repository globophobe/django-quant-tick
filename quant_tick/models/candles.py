from datetime import datetime
from decimal import Decimal

import pandas as pd
from django.db import models
from django.db.models import QuerySet
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    filter_by_timestamp,
    get_existing,
    get_min_time,
    has_timestamps,
    parse_datetime,
)
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField
from .trades import TradeData


class CacheResetMixin:
    """Mixin for candles that support cache reset and sequential aggregation."""

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
            CandleCache.objects.filter(candle=self, timestamp__lt=timestamp_from)
            .only("timestamp", "frequency")
            .last()
        )
        if last_cache:
            ts_from = last_cache.timestamp + pd.Timedelta(f"{last_cache.frequency}min")
            return can_agg and timestamp_from >= ts_from
        return True


class Candle(AbstractCodeName, PolymorphicModel):
    """Candle."""

    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        verbose_name=_("symbol"),
    )
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    def initialize(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> tuple[datetime, datetime, dict]:
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
                t.timestamp + pd.Timedelta(f"{t.frequency}min")
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
                    f"{candle_cache.frequency}min"
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

    def get_trade_data(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        only: list | str | None = None,
    ) -> QuerySet:
        """Get trade data."""
        if isinstance(only, str):
            only = [only]
        only = only or []
        return (
            TradeData.objects.filter(
                symbol__in=self.symbols.all(),
                timestamp__gte=get_min_time(timestamp_from, value="1h"),
                timestamp__lt=timestamp_to,
            )
            .select_related("symbol")
            .only(*["timestamp", "frequency"] + only)
        )

    def get_expected_daily_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> dict:
        """Get expected daily candles."""
        trade_data = self.get_trade_data(timestamp_from, timestamp_to, only="json_data")
        return trade_data.values_list("json_data", flat=True)

    def get_data_frame(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get data frame."""
        trade_data = self.get_trade_data(
            timestamp_from, timestamp_to, only=["symbol"] + list(FileData)
        )
        data_frames = []
        symbols = list(self.symbols.all())
        is_multi_symbol = len(symbols) > 1
        for symbol in symbols:
            target = sorted(
                [obj for obj in trade_data if obj.symbol == symbol],
                key=lambda obj: obj.timestamp,
            )
            dfs = []
            for t in target:
                # Query may contain trade data by minute.
                # Only target timestamps.
                if timestamp_from <= t.timestamp + pd.Timedelta(f"{t.frequency}min"):
                    df = t.get_data_frame(self.json_data["source_data"])
                    if df is not None:
                        dfs.append(df)
            if dfs:
                # Ignore: The behavior of DataFrame concatenation with empty or
                # all-NA entries is deprecated.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    df = pd.concat(dfs)
                # Add exchange and symbol columns if multi-symbol.
                if is_multi_symbol:
                    df.insert(2, "exchange", symbol.exchange)
                    df.insert(3, "symbol", symbol.symbol)
                data_frames.append(df)
        if data_frames:
            df = pd.concat(data_frames)
            sort_values = ["timestamp"]
            if "nanoseconds" in df.columns:
                sort_values.append("nanoseconds")
            df = df.sort_values(sort_values)
            return (
                filter_by_timestamp(df, timestamp_from, timestamp_to)
                .reset_index()
                .drop(columns=["index"])
            )
        else:
            return pd.DataFrame([])

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate?"""
        values = []
        for symbol in self.symbols.all():
            # Trade data may be daily, so timestamp from >= daily timestmap.
            trade_data = TradeData.objects.filter(
                symbol=symbol,
                timestamp__gte=get_min_time(timestamp_from, value="1d"),
                timestamp__lt=timestamp_to,
            )
            # Only target timestamps.
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
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        raise NotImplementedError

    def write_cache(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data: dict | None = None,
        data_frame: DataFrame | None = None,
    ) -> None:
        """Write cache.

        Delete previously saved data.
        """
        queryset = CandleCache.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        )
        queryset.delete()
        delta = timestamp_to - timestamp_from
        # FIXME: Can't define every value in constants.
        frequency = delta.total_seconds() / 60
        CandleCache.objects.create(
            candle=self, timestamp=timestamp_from, frequency=frequency, json_data=data
        )

    def write_data(
        self, timestamp_from: datetime, timestamp_to: datetime, json_data: list[dict]
    ) -> None:
        """Write data.

        Delete previously saved data.
        """
        CandleData.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        ).delete()
        data = []
        for j in json_data:
            timestamp = j.pop("timestamp")
            kwargs = {"timestamp": timestamp, "json_data": j}
            c = CandleData(candle=self, **kwargs)
            data.append(c)
        CandleData.objects.bulk_create(data)

    def get_candle_data(
        self,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        progress: bool = False,
    ) -> DataFrame:
        """Get candle data."""
        queryset = CandleData.objects.filter(candle=self)
        if timestamp_from is not None:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        queryset = queryset.order_by("timestamp")

        data = []
        for idx, obj in enumerate(queryset.iterator(chunk_size=10000)):
            row = {
                "timestamp": obj.timestamp,
                "candle_data_id": obj.pk,
                "bar_index": idx,
                **obj.json_data,
            }
            data.append(row)
        return DataFrame(data)

    def process_data_frame(self, df: DataFrame) -> DataFrame:
        """Process data frame - convert Decimal columns to float."""
        if df.empty:
            return df
        numeric_cols = []
        for col in ["open", "high", "low", "close", "volume", "buyVolume", "notional"]:
            if col in df.columns:
                numeric_cols.append(col)
        for col in numeric_cols:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: float(x) if isinstance(x, Decimal) else x
                )
        return df

    class Meta:
        db_table = "quant_tick_candle"
        verbose_name = _("candle")
        verbose_name_plural = _("candles")


class CandleCache(models.Model):
    """Candle cache."""

    candle = models.ForeignKey(
        "quant_tick.Candle", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(
        _("frequency"), choices=Frequency.choices, db_index=True
    )
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_tick_candle_cache"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle cache")


class CandleData(models.Model):
    """Candle data."""

    candle = models.ForeignKey(
        "quant_tick.Candle", on_delete=models.CASCADE, verbose_name=_("candle")
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_tick_candle_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")
