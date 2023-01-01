import os
from datetime import datetime

import pandas as pd
from django.db import models
from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import (
    filter_by_timestamp,
    get_existing,
    get_missing,
    get_next_time,
)
from quant_candles.querysets import TimeFrameQuerySet
from quant_candles.utils import gettext_lazy as _

from .base import AbstractDataStorage, JSONField
from .symbols import Symbol


def upload_data_to(instance: "TradeData", filename: str) -> str:
    """Upload data to.

    Examples:

    1. trades / coinbase / BTCUSD / raw / 2022-01-01 / 0000.parquet
    2. trades / coinbase / BTCUSD / aggregated / 0 / 2022-01-01 / 0000.parquet
    3. trades / coinbase / BTCUSD / aggregated / 1000 / 2022-01-01 / 0000.parquet
    """
    parts = ["trades", instance.symbol.exchange, instance.symbol.symbol]
    if instance.symbol.should_aggregate_trades:
        parts += ["aggregated", str(instance.symbol.significant_trade_filter)]
    else:
        parts.append("raw")
    parts.append(instance.timestamp.date().isoformat())
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    parts.append(f"{fname}{ext}")
    return "/".join(parts)


class TradeDataQuerySet(TimeFrameQuerySet):
    def get_last_uid(self, symbol: Symbol, timestamp: datetime) -> str:
        """Get last uid."""
        queryset = self.filter(symbol=symbol, timestamp__gte=timestamp)
        obj = queryset.exclude(uid="").order_by("timestamp").first()
        if obj:
            return obj.uid


class TradeData(AbstractDataStorage):
    symbol = models.ForeignKey(
        "quant_candles.Symbol", related_name="aggregated", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    uid = models.CharField(_("uid"), blank=True, max_length=255)
    frequency = models.PositiveIntegerField(
        _("frequency"),
        choices=[
            c for c in Frequency.choices if c[0] in (Frequency.MINUTE, Frequency.HOUR)
        ],
        db_index=True,
    )
    file_data = models.FileField(_("file data"), blank=True, upload_to=upload_data_to)
    json_data = JSONField(_("json data"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = TradeDataQuerySet.as_manager()

    @classmethod
    def write(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        validated: dict,
    ) -> None:
        """Write data."""
        is_first_minute = timestamp_from.time().minute == 0
        is_hourly = timestamp_from == timestamp_to - pd.Timedelta("1h")
        if is_first_minute and is_hourly:
            cls.write_hour(
                symbol,
                timestamp_from,
                timestamp_to,
                data_frame,
                validated,
            )
        else:
            cls.write_minutes(
                symbol,
                timestamp_from,
                timestamp_to,
                data_frame,
                validated,
            )

    @classmethod
    def write_hour(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        validated: dict,
    ) -> None:
        """Write hourly data."""
        params = {
            "symbol": symbol,
            "timestamp": timestamp_from,
            "frequency": Frequency.HOUR,
        }
        try:
            obj = cls.objects.get(**params)
        except cls.DoesNotExist:
            obj = cls(**params)
        finally:
            cls.write_data_frame(obj, data_frame, validated)

    @classmethod
    def write_minutes(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        validated: dict,
    ) -> None:
        """Write minute data."""
        queryset = cls.objects.filter(
            symbol=symbol, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        )
        existing = get_existing(queryset.values("timestamp", "frequency"))
        timestamps = get_missing(timestamp_from, timestamp_to, existing)
        existing = {
            obj.timestamp: obj
            for obj in cls.objects.filter(
                symbol=symbol,
                timestamp__in=timestamps,
                frequency=Frequency.MINUTE,
            )
        }
        for timestamp in timestamps:
            if timestamp in existing:
                obj = existing[timestamp]
            else:
                obj = cls(
                    symbol=symbol,
                    timestamp=timestamp,
                    frequency=Frequency.MINUTE,
                )
            cls.write_data_frame(
                obj,
                filter_by_timestamp(
                    data_frame, timestamp, get_next_time(timestamp, "1t")
                ),
                {timestamp: validated.get(timestamp, None)},
            )

    @classmethod
    def write_data_frame(
        cls,
        obj: models.Model,
        data_frame: DataFrame,
        validated: DataFrame,
    ) -> None:
        """Write data frame.

        Delete previously saved data.
        """
        if obj.pk:
            obj.file_data.delete()
        if len(data_frame):
            obj.uid = data_frame.iloc[0].uid
            obj.file_data = cls.prepare_data(data_frame)
        if len(validated):
            values = validated.values()
            all_true = all([v is True for v in values])
            some_false = True in [isinstance(v, dict) for v in values]
            some_none = None in values
            if all_true:
                obj.ok = True
            elif some_false or some_none:
                if some_false:
                    obj.ok = False
                else:
                    obj.ok = None
                validation_failure = {
                    timestamp.isoformat(): value
                    for timestamp, value in validated.items()
                    if value is None or isinstance(value, dict)
                }
                if validation_failure:
                    obj.json_data = validation_failure
            else:
                raise NotImplementedError
        obj.save()

    class Meta:
        db_table = "quant_candles_trade_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("trade data")
