import os
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
from django.db import models
from django.db.models import Q, QuerySet
from pandas import DataFrame

from quant_werks.constants import Frequency
from quant_werks.lib import (
    aggregate_rows,
    get_current_time,
    get_min_time,
    get_next_time,
    get_range,
    iter_missing,
    iter_timeframe,
)
from quant_werks.utils import gettext_lazy as _

from .base import BaseDataStorage
from .symbols import Symbol


def upload_to(instance: "AggregatedTradeData", filename: str) -> str:
    """Upload to."""
    exchange = instance.symbol.exchange
    upload_symbol = instance.symbol.upload_symbol
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    return f"trades/{exchange}/{upload_symbol}/{date}/{fname}{ext}"


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


class AggregatedTradeData(BaseDataStorage):
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
    data = models.FileField(_("data"), blank=True, upload_to=upload_to)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = AggregatedTradeDataQuerySet.as_manager()

    def get_data_frame(self) -> Optional[DataFrame]:
        """Get data frame."""
        if self.data.name:
            return pd.read_parquet(self.data.file)

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
        validated: Optional[Dict[datetime, bool]] = {},
    ) -> None:
        """Write to database."""
        is_first_minute = timestamp_from.time().minute == 0
        is_hourly = timestamp_from == timestamp_to - pd.Timedelta("1h")
        if is_first_minute and is_hourly:
            cls.write_hour(symbol, timestamp_from, data_frame, validated)
        else:
            cls.write_minutes(
                symbol, timestamp_from, timestamp_to, data_frame, validated
            )

    @classmethod
    def write_hour(
        cls,
        symbol: Symbol,
        timestamp: datetime,
        data_frame: DataFrame,
        validated: Optional[Dict[datetime, bool]] = {},
    ) -> None:
        """Write hourly data to database."""
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "frequency": Frequency.HOUR,
        }
        try:
            obj = cls.objects.get(**params)
        except AggregatedTradeData.DoesNotExist:
            obj = cls(**params)
        finally:
            # Delete previously saved data.
            if obj.pk:
                obj.data.delete()
            if len(data_frame):
                obj.uid = data_frame.iloc[0].uid
                # UID not necessary for hourly data.
                df = data_frame.drop(columns=["uid"])
                obj.data = cls.prepare_data(df)
            else:
                obj.uid = ""

            values = validated.values()
            all_true = all(values)
            some_false = False in values
            some_none = None in values
            if all_true:
                obj.ok = True
            elif some_false:
                obj.ok = False
            elif some_none:
                obj.ok = None
            else:
                raise NotImplementedError
            obj.save()

    @classmethod
    def write_minutes(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        validated: Optional[Dict[datetime, bool]] = {},
    ) -> None:
        """Write minute data to database."""
        timestamps = cls.objects.get_missing(symbol, timestamp_from, timestamp_to)
        existing = {
            obj.timestamp: obj
            for obj in AggregatedTradeData.objects.filter(
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

            if len(data_frame):
                df = data_frame[
                    (data_frame.timestamp >= timestamp)
                    & (data_frame.timestamp < get_next_time(timestamp, "1t"))
                ]
                if len(df):
                    summary = aggregate_rows(
                        df, timestamp=timestamp, nanoseconds=0, is_filtered=True
                    )
                    obj.uid = summary.get("uid", "")
                    obj.data = cls.prepare_data(df)

            obj.ok = validated.get(timestamp, None)
            obj.save()

    class Meta:
        db_table = "quant_werks_aggregated_trade_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("aggregated")
