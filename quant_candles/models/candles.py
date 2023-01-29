import os
from datetime import datetime
from itertools import chain
from typing import List, Optional, Tuple

import pandas as pd
from django.conf import settings
from django.db import models
from django.db.models import Q
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_candles.constants import Frequency
from quant_candles.lib import get_existing, has_timestamps, parse_datetime
from quant_candles.utils import gettext_lazy as _

from .base import AbstractCodeName, AbstractDataStorage, JSONField
from .trades import TradeData


def upload_data_to(instance: "CandleData", filename: str) -> str:
    """Upload to."""
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    prefix = f"candles/{instance.candle.code_name}/{date}"
    return f"{prefix}/{fname}{ext}"


class Candle(AbstractCodeName, PolymorphicModel):
    symbols = models.ManyToManyField(
        "quant_candles.Symbol",
        db_table="quant_candles_candle_symbol",
        verbose_name=_("symbols"),
    )
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    def initialize(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> Tuple[datetime, datetime, Optional[dict]]:
        """Initialize."""
        if self.date_from:
            min_timestamp_from = parse_datetime(self.date_from)
            ts_from = (
                min_timestamp_from
                if timestamp_from < min_timestamp_from
                else timestamp_from
            )
        else:
            ts_from = timestamp_from
        if self.date_to:
            max_timestamp_to = parse_datetime(self.date_to)
            ts_to = (
                max_timestamp_to if timestamp_to < max_timestamp_to else timestamp_to
            )
        else:
            ts_to = timestamp_to
        # Get last cache.
        # Subtract 1 hour, as either Frequency.HOUR or Frequency.MINUTE
        candle_cache = (
            CandleCache.objects.filter(
                candle=self,
                timestamp__gte=ts_from - pd.Timedelta("1h"),
            )
            .order_by("-timestamp")
            .only("timestamp", "json_data")
            .first()
        )
        if candle_cache:
            if not retry:
                timestamp = candle_cache.timestamp
                ts_from = timestamp if timestamp > ts_from else ts_from
            data = candle_cache.json_data
        else:
            data = self.get_initial_cache(ts_from)
        return ts_from, ts_to, data

    def get_initial_cache(self, timestamp: datetime) -> Optional[dict]:
        """Get initial cache."""
        return None

    def get_cache_data(
        self, timestamp: datetime, data: Optional[dict] = None
    ) -> Optional[dict]:
        """Get cache data."""
        return data

    def get_data_frame(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get data frame."""
        trade_data = (
            TradeData.objects.filter(
                symbol__in=self.symbols.all(),
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
            )
            .select_related("symbol")
            .only("symbol", "file_data")
        )
        data_frames = []
        for symbol in self.symbols.all():
            target = sorted(
                [obj for obj in trade_data if obj.symbol == symbol],
                key=lambda obj: obj.timestamp,
            )
            dfs = []
            for t in target:
                df = t.get_data_frame()
                if df is not None:
                    dfs.append(df)
            if dfs:
                df = pd.concat(dfs)
                df.insert(2, "exchange", symbol.exchange)
                df.insert(3, "symbol", symbol.symbol)
                data_frames.append(df)
        if data_frames:
            return (
                pd.concat(data_frames)
                .sort_values(["timestamp", "nanoseconds"])
                .reset_index()
            )
        else:
            return pd.DataFrame([])

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        values = []
        for symbol in self.symbols.all():
            trade_data = TradeData.objects.filter(
                symbol=symbol,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
            )
            existing = get_existing(trade_data.values("timestamp", "frequency"))
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
        self, timestamp_from: datetime, timestamp_to: datetime, limit: int = 1000
    ) -> list:
        """Get data."""
        timeframe_query = Q(timestamp__gte=timestamp_from) & Q(
            timestamp__lte=timestamp_to
        )
        candle_data = (
            CandleData.objects.filter(Q(candle=self) & timeframe_query)
            .only("timestamp", "json_data")
            .order_by("-timestamp")
            .values("timestamp", "json_data")
        )
        candle_data = candle_data[:limit]
        candle_read_only_data = (
            CandleReadOnlyData.objects.filter(Q(candle_id=self.id) & timeframe_query)
            .order_by("-timestamp")
            .values("timestamp", "json_data")
        )
        candle_read_only_data = candle_read_only_data[: limit - candle_data.count()]
        return list(chain(candle_data, candle_read_only_data))

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
        total_minutes = delta.total_seconds() / 60
        frequency = (
            Frequency.HOUR if total_minutes == Frequency.HOUR else Frequency.MINUTE
        )
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
            CandleReadOnlyData.objects.filter(Q(candle_id=self.id) & query).delete()
        else:
            CandleData.objects.filter(Q(candle=self) & query).delete()
        candle_data = []
        for j in json_data:
            timestamp = j.pop("timestamp")
            kwargs = {"timestamp": timestamp, "json_data": j}
            if settings.IS_LOCAL:
                c = CandleReadOnlyData(candle_id=self.id, **kwargs)
            else:
                c = CandleData(candle=self, **kwargs)
            candle_data.append(c)
        if settings.IS_LOCAL:
            CandleReadOnlyData.objects.bulk_create(candle_data)
        else:
            CandleData.objects.bulk_create(candle_data)

    class Meta:
        db_table = "quant_candles_candle"
        verbose_name = _("candle")
        verbose_name_plural = _("candles")


class CandleCache(AbstractDataStorage):
    candle = models.ForeignKey(
        "quant_candles.Candle", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(
        _("frequency"), choices=Frequency.choices, db_index=True
    )
    json_data = JSONField(_("json data"), null=True)

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
        "quant_candles.Candle", on_delete=models.CASCADE, verbose_name=_("candle")
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
