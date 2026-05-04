from datetime import datetime

import pandas as pd
from django.db import models, transaction
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.lib import (
    normalize_timestamp_data_frame,
    to_decimal_or_none,
    to_utc_datetime,
)

from .base import (
    BigDecimalField,
    JSONField,
    get_model_json_data,
)
from .symbols import Symbol


class ExchangeCandleDataQuerySet(QuerySet):
    """QuerySet helpers for exchange candles."""

    def in_range(
        self,
        symbol: Symbol,
        frequency: int,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> QuerySet:
        return self.filter(
            symbol=symbol,
            frequency=frequency,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )


class ExchangeCandleData(models.Model):
    """Exchange candles, not candles aggregated from trade data."""

    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        related_name="exchange_candle_data",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(_("frequency"), db_index=True)
    open = BigDecimalField(_("open"))
    high = BigDecimalField(_("high"))
    low = BigDecimalField(_("low"))
    close = BigDecimalField(_("close"))
    volume = BigDecimalField(_("volume"), null=True, blank=True)
    notional = BigDecimalField(_("notional"), null=True, blank=True)
    json_data = JSONField(_("json data"), default=dict)
    objects = ExchangeCandleDataQuerySet.as_manager()

    @classmethod
    def write(
        cls,
        symbol: Symbol,
        frequency: int,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
    ) -> None:
        """Replace exchange candles for half-open timestamp range."""
        rows = []
        frame = normalize_timestamp_data_frame(data_frame)
        for row in frame.to_dict("records"):
            timestamp = to_utc_datetime(row["timestamp"])
            if not timestamp_from <= timestamp < timestamp_to:
                continue
            json_data = get_model_json_data(
                row,
                {
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "notional",
                },
            )
            rows.append(
                cls(
                    symbol=symbol,
                    timestamp=timestamp,
                    frequency=frequency,
                    open=to_decimal_or_none(row["open"]),
                    high=to_decimal_or_none(row["high"]),
                    low=to_decimal_or_none(row["low"]),
                    close=to_decimal_or_none(row["close"]),
                    volume=to_decimal_or_none(row.get("volume")),
                    notional=to_decimal_or_none(row.get("notional")),
                    json_data=json_data,
                )
            )

        with transaction.atomic():
            cls.objects.in_range(
                symbol,
                frequency,
                timestamp_from,
                timestamp_to,
            ).delete()
            cls.objects.bulk_create(rows)

    def coverage_end(self) -> datetime:
        return self.timestamp + pd.Timedelta(f"{self.frequency}min")

    def to_row(
        self,
        *,
        include_volume: bool = True,
        include_notional: bool = True,
    ) -> dict:
        row = {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }
        if include_volume:
            row["volume"] = self.volume
        if include_notional:
            row["notional"] = self.notional
        if self.json_data:
            row.update(self.json_data)
        return row

    class Meta:
        db_table = "quant_tick_exchange_candle_data"
        ordering = ("timestamp",)
        unique_together = (("symbol", "timestamp", "frequency"),)
        verbose_name = verbose_name_plural = _("exchange candle data")
