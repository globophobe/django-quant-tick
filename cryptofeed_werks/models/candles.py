from datetime import datetime
from decimal import Decimal
from typing import Dict, Generator, List, Optional, Tuple

import pandas as pd
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

from .aggregated_trades import AggregatedTrade
from .base import big_decimal
from .symbols import Future, Symbol

ZERO = Decimal("0")


class Candle(models.Model):
    symbol = models.ForeignKey(
        "cryptofeed_werks.Symbol",
        related_name="candles",
        on_delete=models.CASCADE,
    )
    future = models.ForeignKey(
        "cryptofeed_werks.Future",
        related_name="candles",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        verbose_name=_("future"),
    )
    uid = models.CharField(_("uid"), blank=True, max_length=255)
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    open = big_decimal("open", null=True)
    high = big_decimal("high", null=True)
    low = big_decimal("low", null=True)
    close = big_decimal("close", null=True)
    buy_volume = big_decimal("buy volume", null=True)
    volume = big_decimal("volume", null=True)
    buy_notional = big_decimal("buy notional", null=True)
    notional = big_decimal("notional", null=True)
    buy_ticks = models.PositiveIntegerField(_("buy ticks"), null=True)
    ticks = models.PositiveIntegerField(_("ticks"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)

    @classmethod
    def iter_all(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        future: Future = None,
        reverse: bool = True,
        retry: bool = False,
    ) -> Generator[Tuple[datetime, datetime], None, None]:
        """Iter all, by days in 1 hour chunks, further chunked by 1m intervals.

        1 day -> 24 hours -> 60 minutes or 10 minutes, etc.
        """
        now = get_current_time()
        max_timestamp_to = get_min_time(now, value="1t")
        for daily_timestamp_from, daily_timestamp_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=reverse
        ):
            # Query for daily.
            daily_existing = cls.get_existing(
                symbol,
                daily_timestamp_from,
                daily_timestamp_to,
                future=future,
                retry=retry,
            )
            daily_delta = daily_timestamp_to - daily_timestamp_from
            daily_expected = int(daily_delta.total_seconds() / 60)
            if daily_existing.count() < daily_expected:
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
                            end = (
                                max_timestamp_to
                                if end_time > max_timestamp_to
                                else end_time
                            )
                            yield start_time, end

    @classmethod
    def get_missing(
        cls,
        symbol: Symbol,
        timestamp_from: datetime,
        timestamp_to: datetime,
        future: Future = None,
    ) -> List[datetime]:
        """Get missing."""
        existing = cls.get_existing(symbol, timestamp_from, timestamp_to, future=future)
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
        future: Future = None,
        retry: bool = False,
    ) -> List[datetime]:
        """Existing."""
        queryset = cls.objects.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )
        if future:
            queryset = queryset.filter(future=future)
        if retry:
            queryset = queryset.exclude(ok=False)
        return queryset.values_list("timestamp", flat=True)

    @classmethod
    def get_last_uid(
        cls, symbol: Symbol, timestamp: datetime, future: Future = None
    ) -> str:
        """Get last uid."""
        queryset = cls.objects.filter(symbol=symbol, timestamp__gt=timestamp)
        if future:
            queryset = queryset.filter(future=future)
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
        future: Future = None,
        ok: Optional[bool] = True,
    ) -> None:
        """Write candles to database."""
        timestamps = cls.get_missing(
            symbol, timestamp_from, timestamp_to, future=future
        )
        candles = {
            candle.timestamp: candle
            for candle in cls.objects.filter(
                symbol=symbol, future=future, timestamp__in=timestamps
            )
        }
        for timestamp in timestamps:
            if len(data_frame):
                df, data = cls.aggregate(timestamp, data_frame, "1t")
            else:
                df = pd.DataFrame([])
                data = {}

            if timestamp in candles:
                candle = candles[timestamp]
            else:
                candle = Candle(symbol=symbol, future=future, timestamp=timestamp)
            # Update candle
            candle.uid = data.get("uid", "")
            candle.open = data.get("open")
            candle.high = data.get("high")
            candle.low = data.get("low")
            candle.close = data.get("close")
            candle.buy_volume = data.get("buyVolume")
            candle.volume = data.get("volume")
            candle.buy_notional = data.get("buyNotional")
            candle.notional = data.get("notional")
            candle.buy_ticks = data.get("buyTicks")
            candle.ticks = data.get("ticks")
            candle.ok = validated.get(timestamp, False)
            candle.save()

            # Delete previous aggregated trades
            AggregatedTrade.objects.filter(candle=candle).delete()

            for row in df.itertuples():
                AggregatedTrade.objects.create(
                    candle=candle,
                    timestamp=row.timestamp,
                    nanoseconds=row.nanoseconds,
                    price=row.price,
                    volume=row.volume if not pd.isnull(row.volume) else None,
                    notional=row.notional if not pd.isnull(row.notional) else None,
                    tick_rule=row.tickRule if not pd.isnull(row.tickRule) else None,
                    ticks=row.ticks if not pd.isnull(row.ticks) else None,
                    high=row.high,
                    low=row.low,
                    total_buy_volume=row.totalBuyVolume,
                    total_volume=row.totalVolume,
                    total_buy_notional=row.totalBuyNotional,
                    total_notional=row.totalNotional,
                    total_buy_ticks=row.totalBuyTicks,
                    total_ticks=row.totalTicks,
                )

    @classmethod
    def aggregate(
        cls,
        timestamp: datetime,
        data_frame: DataFrame,
        window: str = "1t",
        is_filtered: bool = True,
    ) -> None:
        """Aggregate candles from data_frame."""
        df = data_frame[
            (data_frame.timestamp >= timestamp)
            & (data_frame.timestamp < get_next_time(timestamp, window))
        ]
        data = {}
        if len(df):
            data = aggregate_rows(
                df, timestamp=timestamp, nanoseconds=0, is_filtered=is_filtered
            )
        return df, data

    def __str__(self) -> str:
        exchange = self.symbol.get_exchange_display()
        symbol = self.symbol.symbol_display
        timestamp = self.timestamp.replace(tzinfo=None).isoformat()
        return f"{exchange} {symbol} {timestamp} {self.ok}"

    class Meta:
        db_table = "cryptofeed_werks_candle"
        ordering = ("timestamp",)
        verbose_name = _("candle")
        verbose_name_plural = _("candles")
