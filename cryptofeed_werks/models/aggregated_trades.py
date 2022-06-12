import os
from datetime import datetime
from io import BytesIO
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
from django.core.files.base import ContentFile
from django.db import models
from pandas import DataFrame

from cryptofeed_werks.lib import (
    aggregate_rows,
    get_current_time,
    get_min_time,
    get_next_time,
    get_range,
    iter_missing,
    iter_timeframe,
)
from cryptofeed_werks.utils import gettext_lazy as _

from .symbols import Symbol


def upload_to(instance, filename):
    """Upload to."""
    exchange = instance.symbol.exchange
    symbol = instance.symbol.symbol
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    return f"{exchange}/{symbol}/{date}/{fname}{ext}"


class AggregatedTradeData(models.Model):
    symbol = models.ForeignKey(
        "cryptofeed_werks.Symbol",
        related_name="aggregated_trade_data",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    uid = models.CharField(_("uid"), blank=True, max_length=255)
    data = models.FileField(_("data"), blank=True, upload_to=upload_to)
    stats = models.JSONField(_("stats"), null=True, blank=True)
    is_hourly = models.BooleanField(
        _("hourly"), null=True, default=False, db_index=True
    )
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)

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
            daily_existing = cls.get_existing(
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
    def get_missing(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> List[datetime]:
        """Get missing."""
        existing = cls.get_existing(symbol, timestamp_from, timestamp_to)
        # Result from get_range may include timestamp_to,
        # which will never be part of data_frame
        if timestamp_to > timestamp_from:
            timestamp_to -= pd.Timedelta("1t")
        return [
            timestamp
            for timestamp in get_range(timestamp_from, timestamp_to)
            if timestamp not in existing
        ]

    @classmethod
    def get_existing(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> List[datetime]:
        """Get existing."""
        queryset = cls.objects.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )
        if retry:
            queryset = queryset.exclude(ok=False)
        # List of 1m timestamps.
        result = []
        for item in queryset.values("timestamp", "is_hourly"):
            timestamp = item["timestamp"]
            if item["is_hourly"]:
                result += [timestamp + pd.Timedelta(index) for index in range(60)]
            else:
                result.append(timestamp)
        return sorted(result)

    @classmethod
    def get_last_uid(cls, symbol: Symbol, timestamp: datetime) -> str:
        """Get last uid."""
        queryset = cls.objects.filter(symbol=symbol, timestamp__gte=timestamp)
        obj = queryset.exclude(uid="").order_by("timestamp").first()
        if obj:
            return obj.uid

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
        params = {"symbol": symbol, "timestamp": timestamp, "is_hourly": True}
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
        timestamps = cls.get_missing(symbol, timestamp_from, timestamp_to)
        existing = {
            obj.timestamp: obj
            for obj in AggregatedTradeData.objects.filter(
                symbol=symbol, timestamp__in=timestamps, is_hourly=False
            )
        }
        for timestamp in timestamps:
            if timestamp in existing:
                obj = existing[timestamp]
            else:
                obj = cls(symbol=symbol, timestamp=timestamp, is_hourly=False)

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

    @classmethod
    def prepare_data(cls, data_frame: DataFrame) -> ContentFile:
        """Prepare data, exclude uid."""
        buffer = BytesIO()
        data_frame.to_parquet(buffer, engine="auto", compression="snappy")
        return ContentFile(buffer.getvalue(), "data.parquet")

    @property
    def data_frame(self) -> DataFrame:
        """Load data frame."""
        if self.data.name:
            return pd.read_parquet(self.data.open())

    class Meta:
        db_table = "cryptofeed_werks_aggregated_trade_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("aggregated")
