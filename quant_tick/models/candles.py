import logging
from collections.abc import Generator, Iterator
from datetime import datetime
from decimal import Decimal

import pandas as pd
from django.db import models
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    aggregate_candle,
    filter_by_timestamp,
    get_current_time,
    get_min_time,
    get_next_cache,
    merge_cache,
    parse_datetime,
)
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField
from .trades import TradeData

logger = logging.getLogger(__name__)


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
        first_trade = (
            TradeData.objects.filter(symbol=self.symbol).only("timestamp").first()
        )
        if first_trade and ts_from < first_trade.timestamp:
            ts_from = first_trade.timestamp
        last_trade = (
            TradeData.objects.filter(symbol=self.symbol)
            .only("timestamp", "frequency")
            .last()
        )
        if last_trade:
            max_ts_to = min(
                last_trade.timestamp + pd.Timedelta(f"{last_trade.frequency}min"),
                ts_to,
            )
        else:
            max_ts_to = ts_to
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
            cache_end = candle_cache.timestamp + pd.Timedelta(
                f"{candle_cache.frequency}min"
            )
            if not retry:
                # Gap: requested start is after cache end
                if ts_from > cache_end:
                    ts_from = ts_to  # Force no processing
                else:
                    ts_from = cache_end
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

    def candles(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> None:
        """Candles."""
        min_ts_from, max_ts_to, cache_data = self.initialize(
            timestamp_from, timestamp_to, retry
        )
        if retry:
            self.on_retry(min_ts_from, max_ts_to)
        for ts_from, ts_to, trade_data in self.iter_all(min_ts_from, max_ts_to):
            df = self.get_data_frame(ts_from, ts_to, trade_data)
            cache_data = self.get_cache_data(ts_from, cache_data)
            data, cache_data = self.aggregate(ts_from, ts_to, df, cache_data)
            self.write_cache(ts_from, ts_to, cache_data)
            self.write_data(ts_from, ts_to, data)
            ts = ts_to.replace(tzinfo=None)
            logger.info(f"Candle {self}: {ts}")

    def on_retry(self, timestamp_from: datetime, timestamp_to: datetime) -> None:
        """On retry."""
        CandleCache.objects.filter(candle=self, timestamp__gte=timestamp_from).delete()
        CandleData.objects.filter(candle=self, timestamp__gte=timestamp_from).delete()

    def iter_all(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> Generator[tuple[datetime, datetime, TradeData], None, None]:
        """Iterate all."""
        max_ts_to = get_min_time(get_current_time(), value="1min")
        ts_to = min(timestamp_to, max_ts_to)
        if timestamp_from < ts_to:
            queryset = (
                TradeData.objects.filter(
                    symbol=self.symbol,
                    timestamp__gte=get_min_time(timestamp_from, value="1d"),
                    timestamp__lt=ts_to,
                )
                .order_by("timestamp")
                .only("timestamp", "frequency", *list(FileData))
            )
            prev_end = None
            for obj in queryset.iterator(chunk_size=100):
                obj_from = obj.timestamp
                obj_to = obj.timestamp + pd.Timedelta(f"{obj.frequency}min")
                # Stop on gap between TradeData
                if prev_end is not None and obj_from != prev_end:
                    break
                prev_end = obj_to
                # Clamp to requested range
                td_from = max(obj_from, timestamp_from)
                td_to = min(obj_to, ts_to, max_ts_to)
                if td_from < td_to and self.can_aggregate(td_from, td_to):
                    yield td_from, td_to, obj

    def get_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        trade_data: TradeData,
    ) -> DataFrame:
        """Get data frame from trade data."""
        df = trade_data.get_data_frame(self.json_data["source_data"])
        if df is not None and len(df):
            return filter_by_timestamp(df, timestamp_from, timestamp_to).reset_index(
                drop=True
            )
        return pd.DataFrame([])

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        return True

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        raise NotImplementedError

    def _preprocess_data(self, data_frame: DataFrame, cache_data: dict) -> DataFrame:
        """Preprocess data before iteration. Override in mixins."""
        return data_frame

    def _build_candle(self, df: DataFrame, timestamp: datetime | None = None) -> dict:
        """Build candle dict. Override in mixins to add extensions."""
        return aggregate_candle(
            df,
            timestamp=timestamp,
            distribution_stats=self.json_data.get("distribution_stats", False),
        )

    def _merge_cache(self, prev: dict, curr: dict) -> dict:
        """Merge cached candle with current. Override to add bucket stats merging."""
        return merge_cache(prev, curr)

    def _get_next_cache(
        self,
        df: DataFrame,
        cache_data: dict,
        timestamp: datetime | None = None,
    ) -> dict:
        """Cache incomplete candle. Override to include bucket stats."""
        return get_next_cache(df, cache_data, timestamp=timestamp)

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
        is_complete: bool = True,
    ) -> Iterator[dict]:
        """Get candle data as iterator of dicts."""
        queryset = CandleData.objects.filter(candle=self)
        if timestamp_from is not None:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        queryset = queryset.order_by("timestamp")

        for idx, obj in enumerate(queryset.iterator(chunk_size=10000)):
            yield {
                "timestamp": obj.timestamp,
                "candle_data_id": obj.pk,
                "bar_index": idx,
                **obj.json_data,
            }

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
