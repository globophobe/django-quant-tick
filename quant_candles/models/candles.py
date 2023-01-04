import os
from datetime import datetime
from io import BytesIO
from itertools import chain
from typing import List, Optional, Tuple

import pandas as pd
from django.conf import settings
from django.db import models
from django.db.models import Q
from pandas import DataFrame
from polymorphic.models import PolymorphicModel

from quant_candles.constants import Frequency
from quant_candles.lib import get_existing, has_timestamps
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
        verbose_name=_("symbol"),
    )
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    def get_initial_cache(self, **kwargs) -> Tuple[Optional[dict], Optional[BytesIO]]:
        """Get initial cache."""
        raise NotImplementedError

    def get_last_cache(
        self, timestamp: datetime
    ) -> Tuple[Optional[dict], Optional[BytesIO]]:
        """Get cache."""
        candle_cache = (
            CandleCache.objects.filter(candle=self, timestamp__lt=timestamp)
            .only("timestamp", "json_data")
            .first()
        )
        if candle_cache:
            file_data = (
                candle_cache.get_file_data() if candle_cache.file_data.name else None
            )
            return candle_cache.json_data, file_data
        return None, None

    def get_cache(
        self,
        timestamp: datetime,
        json_data: Optional[dict] = None,
        file_data: Optional[BytesIO] = None,
    ) -> Tuple[Optional[dict], Optional[BytesIO]]:
        """Get cache."""
        if json_data is None and file_data is None:
            return self.get_last_cache(timestamp)
        return json_data, file_data

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate."""
        values = []
        for symbol in self.symbols.all():
            trade_data = (
                TradeData.objects.filter(
                    symbol=symbol,
                    timestamp__gte=timestamp_from,
                    timestamp__lt=timestamp_to,
                )
                .only("timestamp", "frequency")
                .values("timestamp", "frequency")
            )
            existing = get_existing(trade_data)
            values.append(has_timestamps(timestamp_from, timestamp_to, existing))
        return all(values)

    def aggregate(
        self,
        data_frame: DataFrame,
        json_data: dict,
        file_data: Optional[BytesIO] = None,
    ) -> Tuple[list, dict]:
        """Aggregate."""
        raise NotImplementedError

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
            .only("symbol")
        )
        data_frames = []
        for t in trade_data:
            data_frame = t.get_data_frame()
            if data_frame is not None:
                data_frame.insert(2, "exchange", t.symbol.exchange)
                data_frame.insert(3, "symbol", t.symbol.api_symbol)
                data_frames.append(data_frame)
        if data_frames:
            return (
                pd.concat(data_frames)
                .sort_values(["timestamp", "nanoseconds"])
                .reset_index()
            )
        else:
            return pd.DataFrame([])

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
        json_data: Optional[dict] = None,
        file_data: Optional[BytesIO] = None,
    ):
        """Write cache.

        Delete previously saved data.
        """
        queryset = CandleCache.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_from
        )
        queryset.delete()
        delta = timestamp_to - timestamp_from
        frequency = (
            Frequency.HOUR
            if delta.total_seconds() == Frequency.HOUR
            else Frequency.MINUTE
        )
        CandleCache.objects.create(
            candle=self,
            timestamp=timestamp_from,
            frequency=frequency,
            json_data=json_data,
            file_data=file_data,
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
    file_data = models.FileField(_("file data"), blank=True, upload_to=upload_data_to)
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
