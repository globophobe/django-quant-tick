import logging
import re
from collections.abc import Generator, Iterator
from datetime import datetime
from decimal import Decimal

import pandas as pd
from django.db import models, transaction
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_tick.constants import FileData
from quant_tick.lib import (
    aggregate_candle,
    filter_by_timestamp,
    get_current_time,
    get_min_time,
    get_next_cache,
    merge_cache,
    parse_datetime,
)

from .base import AbstractCodeName, BigDecimalField, JSONField
from .trades import TradeData

logger = logging.getLogger(__name__)


def camel_to_snake(value: str) -> str:
    """Convert camelCase to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


class Candle(AbstractCodeName, PolymorphicModel):
    """Base candle model."""

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
        """Clamp the requested range to available trade and cache data."""
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
        """Hook for the initial cache state."""
        return {}

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Hook for per-slice cache updates."""
        return data

    def candles(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> None:
        """Aggregate and persist candles over a time range."""
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
        """Delete cached and persisted candle rows before retrying."""
        with transaction.atomic():
            CandleCache.objects.filter(
                candle=self, timestamp__gte=timestamp_from
            ).delete()
            CandleData.objects.filter(
                candle=self, timestamp__gte=timestamp_from
            ).delete()

    def iter_all(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> Generator[tuple[datetime, datetime, TradeData], None, None]:
        """Iterate TradeData slices in the requested range without gaps."""
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
                    logger.warning(
                        "Candle %s stopped on TradeData gap: expected next timestamp %s but found %s",
                        self,
                        prev_end,
                        obj_from,
                    )
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
        """Load and clip one TradeData slice."""
        df = trade_data.get_data_frame(self.json_data["source_data"])
        if df is not None and len(df):
            return filter_by_timestamp(df, timestamp_from, timestamp_to).reset_index(
                drop=True
            )
        return pd.DataFrame([])

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Whether this trade-data slice should be processed."""
        return True

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate trade data into completed candles and cache state."""
        raise NotImplementedError

    def _preprocess_data(self, data_frame: DataFrame, cache_data: dict) -> DataFrame:
        """Preprocess data before iteration. Override in mixins."""
        return data_frame

    def _aggregate_candle(
        self, df: DataFrame, timestamp: datetime | None = None
    ) -> dict:
        """Aggregate one candle payload from a trade slice."""
        return aggregate_candle(
            df,
            timestamp=timestamp,
            min_volume_exponent=self.json_data.get("min_volume_exponent"),
            min_notional_exponent=self.json_data.get("min_notional_exponent"),
        )

    def _merge_cache(self, prev: dict, curr: dict) -> dict:
        """Hook for merging cached candle state into a finished candle."""
        return merge_cache(prev, curr)

    def _get_next_cache(
        self,
        df: DataFrame,
        cache_data: dict,
        timestamp: datetime | None = None,
    ) -> dict:
        """Hook for building the next incomplete-cache payload."""
        return get_next_cache(
            df,
            cache_data,
            timestamp=timestamp,
            min_volume_exponent=self.json_data.get("min_volume_exponent"),
            min_notional_exponent=self.json_data.get("min_notional_exponent"),
        )

    def write_cache(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data: dict | None = None,
        data_frame: DataFrame | None = None,
    ) -> None:
        """Replace cache rows in the requested window."""
        with transaction.atomic():
            queryset = CandleCache.objects.filter(
                candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
            )
            queryset.delete()
            span_minutes = int((timestamp_to - timestamp_from).total_seconds() / 60)
            CandleCache.objects.create(
                candle=self,
                timestamp=timestamp_from,
                frequency=span_minutes,
                json_data=data,
            )

    def write_data(
        self, timestamp_from: datetime, timestamp_to: datetime, json_data: list[dict]
    ) -> None:
        """Replace candle rows in the requested window."""
        with transaction.atomic():
            CandleData.objects.filter(
                candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
            ).delete()
            data = []
            for j in json_data:
                data.append(CandleData.objects.from_data(self, j))
            CandleData.objects.bulk_create(data)

    def get_candle_data(
        self,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        is_complete: bool = True,
    ) -> Iterator[dict]:
        """Yield stored candle rows as plain dicts."""
        queryset = CandleData.objects.filter(candle=self)
        if timestamp_from is not None:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        queryset = queryset.order_by("timestamp")

        for obj in queryset.iterator(chunk_size=10000):
            payload = {}
            for field_name in CandleData.DATA_FIELDS:
                value = getattr(obj, field_name)
                if field_name == "incomplete":
                    if value:
                        payload[field_name] = True
                    continue
                if value is not None:
                    payload[field_name] = value
            if obj.extra_data:
                payload.update(obj.extra_data)
            yield {
                "timestamp": obj.timestamp,
                **payload,
            }

    def process_data_frame(self, df: DataFrame) -> DataFrame:
        """Convert Decimal-valued columns to floats."""
        if df.empty:
            return df
        for col in df.columns:
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
    """Cached incomplete candle state keyed by covered minutes."""

    candle = models.ForeignKey(
        "quant_tick.Candle", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(_("frequency"), db_index=True)
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_tick_candle_cache"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle cache")


class CandleDataManager(models.Manager):
    """Manager for constructing candle rows from aggregated data."""

    def from_data(self, candle: Candle, data: dict) -> "CandleData":
        """Build one CandleData row from aggregated candle output."""
        remaining = dict(data)
        kwargs = {
            "candle": candle,
            "timestamp": remaining.pop("timestamp"),
            "extra_data": {},
        }
        for key in list(remaining):
            field_name = camel_to_snake(key)
            if field_name in self.model.DATA_FIELDS:
                kwargs[field_name] = remaining.pop(key)
        if remaining:
            kwargs["extra_data"] = remaining
        return self.model(**kwargs)


class CandleData(models.Model):
    """Stored aggregated candle row."""

    DATA_FIELDS = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "notional",
        "buy_notional",
        "ticks",
        "buy_ticks",
        "realized_variance",
        "incomplete",
    )

    candle = models.ForeignKey(
        "quant_tick.Candle", on_delete=models.CASCADE, verbose_name=_("candle")
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    open = BigDecimalField(_("open"), null=True, blank=True)
    high = BigDecimalField(_("high"), null=True, blank=True)
    low = BigDecimalField(_("low"), null=True, blank=True)
    close = BigDecimalField(_("close"), null=True, blank=True)
    volume = BigDecimalField(_("volume"))
    buy_volume = BigDecimalField(_("buy volume"))
    notional = BigDecimalField(_("notional"))
    buy_notional = BigDecimalField(_("buy notional"))
    ticks = models.PositiveIntegerField(_("ticks"))
    buy_ticks = models.PositiveIntegerField(_("buy ticks"))
    realized_variance = BigDecimalField(
        _("realized variance"),
    )
    incomplete = models.BooleanField(_("incomplete"), default=False)
    extra_data = JSONField(_("extra data"), default=dict)
    objects = CandleDataManager()

    class Meta:
        db_table = "quant_tick_candle_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")
