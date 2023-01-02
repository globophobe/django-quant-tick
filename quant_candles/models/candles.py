import os
from datetime import datetime
from itertools import chain
from typing import Tuple

import pandas as pd
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.db.models import Q
from pandas import DataFrame

from quant_candles.constants import Frequency
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


class Candle(AbstractCodeName):
    symbols = models.ManyToManyField(
        "quant_candles.Symbol",
        db_table="quant_candles_candle_symbol",
        verbose_name=_("symbol"),
    )
    content_type = models.ForeignKey(ContentType, on_delete=models.SET_NULL, null=True)
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    @classmethod
    def get_content_types(cls) -> list:
        """Get content types."""
        pks = [
            ctype
            for ctype, model_class in [
                (ctype, ctype.model_class()) for ctype in ContentType.objects.all()
            ]
            if issubclass(model_class, Candle) and model_class is not Candle
        ]
        return ContentType.objects.filter(pk__in=pks)

    def get_initial_cache(self, **kwargs) -> dict:
        """Get initial cache."""
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
            data_frame.insert(2, "exchange", t.symbol.exchange)
            data_frame.insert(3, "symbol", t.symbol.api_symbol)
            data_frames.append(data_frame)
        return (
            pd.concat(data_frames)
            .sort_values(["timestamp", "nanoseconds"])
            .reset_index()
        )

    def aggregate(self, data_frame: DataFrame, cache: dict) -> Tuple[list, dict]:
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
