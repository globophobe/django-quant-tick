import os
from datetime import datetime
from typing import Generator, List, Optional, Tuple

import pandas as pd
from django.db import models
from django.db.models import Q, QuerySet
from pandas import DataFrame

from quant_werks.constants import Frequency
from quant_werks.lib import (
    filter_by_timestamp,
    get_current_time,
    get_min_time,
    get_next_time,
    get_range,
    iter_missing,
    iter_timeframe,
)
from quant_werks.utils import gettext_lazy as _

from .base import AbstractDataStorage
from .symbols import Symbol


def upload_data_to(instance: "AggregatedTradeData", filename: str) -> str:
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


class AggregatedTradeDataQuerySet(models.QuerySet):
    def filter_by_timestamp(
        self,
        timestamp_from: Optional[datetime] = None,
        timestamp_to: Optional[datetime] = None,
    ) -> QuerySet:
        """Filter by timestamp."""
        q = Q()
        if timestamp_from:
            q |= Q(timestamp__gte=timestamp_from)
        if timestamp_to:
            q |= Q(timestamp__lt=timestamp_to)
        return self.filter(q)

    def get_missing(
        self, symbol: Symbol, timestamp_from: datetime, timestamp_to: datetime
    ) -> List[datetime]:
        """Get missing."""
        existing = self.get_existing(symbol, timestamp_from, timestamp_to)
        # Result from get_range may include timestamp_to,
        # which will never be part of data_frame
        if timestamp_to > timestamp_from:
            timestamp_to -= pd.Timedelta("1t")
        return [
            timestamp
            for timestamp in get_range(timestamp_from, timestamp_to)
            if timestamp not in existing
        ]

    def get_existing(
        self,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> List[datetime]:
        """Get existing."""
        queryset = self.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )
        if retry:
            queryset = queryset.exclude(ok=False)
        # List of 1m timestamps.
        result = []
        for item in queryset.values("timestamp", "frequency"):
            timestamp = item["timestamp"]
            frequency = item["frequency"]
            if frequency == Frequency.HOUR:
                result += [timestamp + pd.Timedelta(index) for index in range(60)]
            else:
                result.append(timestamp)
        return sorted(result)

    def get_last_uid(self, symbol: Symbol, timestamp: datetime) -> str:
        """Get last uid."""
        queryset = self.filter(symbol=symbol, timestamp__gte=timestamp)
        obj = queryset.exclude(uid="").order_by("timestamp").first()
        if obj:
            return obj.uid


class AggregatedTradeData(AbstractDataStorage):
    symbol = models.ForeignKey(
        "quant_werks.Symbol", related_name="aggregated", on_delete=models.CASCADE
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
    json_data = models.JSONField(_("json data"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = AggregatedTradeDataQuerySet.as_manager()

    @classmethod
    def iter_all(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        reverse: bool = True,
        retry: bool = False,
    ) -> Generator[Tuple[datetime, datetime], None, None]:
        """Iter all, by days in 1 hour chunks, further chunked by 1m intervals.

        1 day -> 24 hours -> 60 minutes or 10 minutes, etc.
        """
        now = get_current_time()
        max_timestamp_to = get_min_time(now, value="1t")
        for daily_timestamp_from, daily_timestamp_to, daily_existing in cls.iter_days(
            symbol, timestamp_from, timestamp_to, reverse=reverse, retry=retry
        ):
            for hourly_timestamp_from, hourly_timestamp_to in cls.iter_hours(
                daily_timestamp_from,
                daily_timestamp_to,
                max_timestamp_to,
                daily_existing,
                reverse=reverse,
                retry=retry,
            ):
                yield hourly_timestamp_from, hourly_timestamp_to

    @classmethod
    def iter_days(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        reverse: bool = True,
        retry: bool = False,
    ):
        """Iter days."""
        for daily_timestamp_from, daily_timestamp_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=reverse
        ):
            # Query for daily.
            daily_existing = cls.objects.get_existing(
                symbol, daily_timestamp_from, daily_timestamp_to, retry=retry
            )
            daily_delta = daily_timestamp_to - daily_timestamp_from
            daily_expected = int(daily_delta.total_seconds() / 60)
            if len(daily_existing) < daily_expected:
                yield daily_timestamp_from, daily_timestamp_to, daily_existing

    @classmethod
    def iter_hours(
        cls,
        daily_timestamp_from: datetime,
        daily_timestamp_to: datetime,
        max_timestamp_to: datetime,
        daily_existing: List[datetime],
        reverse: bool = True,
        retry: bool = False,
    ):
        """Iter hours."""
        for hourly_timestamp_from, hourly_timestamp_to in iter_timeframe(
            daily_timestamp_from,
            daily_timestamp_to,
            value="1h",
            reverse=reverse,
        ):
            # List comprehension for hourly.
            hourly_existing = [
                timestamp
                for timestamp in daily_existing
                if timestamp >= hourly_timestamp_from
                and timestamp < hourly_timestamp_to
            ]
            hourly_delta = hourly_timestamp_to - hourly_timestamp_from
            hourly_expected = int(hourly_delta.total_seconds() / 60)
            if len(hourly_existing) < hourly_expected:
                for start_time, end_time in iter_missing(
                    hourly_timestamp_from,
                    hourly_timestamp_to,
                    hourly_existing,
                    reverse=reverse,
                ):
                    end = max_timestamp_to if end_time > max_timestamp_to else end_time
                    yield start_time, end

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
            obj.save()

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
        timestamps = cls.objects.get_missing(symbol, timestamp_from, timestamp_to)
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
            obj.save()

    @classmethod
    def write_data_frame(
        cls,
        obj: models.Model,
        data_frame: DataFrame,
        validated: DataFrame,
    ) -> None:
        # Delete previously saved data.
        if obj.pk:
            obj.file_data.delete()
        if len(data_frame):
            obj.uid = data_frame.iloc[0].uid
            obj.file_data = cls.prepare_data(data_frame)
        if len(validated):
            values = validated.values()
            all_true = all([v is True for v in values])
            some_false = len([isinstance(v, dict) for v in values])
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

    class Meta:
        db_table = "quant_werks_aggregated_trade_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("aggregated trade data")
