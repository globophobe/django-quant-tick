from datetime import datetime
from typing import Generator, List, Tuple

from django.db import models

from quant_candles.lib import (
    get_current_time,
    get_existing,
    get_min_time,
    iter_missing,
    iter_timeframe,
)
from quant_candles.models import Candle, CandleCache, Symbol, TradeData


class BaseTimeFrameIterator:
    def __init__(self, obj: models.Model) -> None:
        self.obj = obj
        self.reverse = None

    def get_max_timestamp_to(self) -> datetime:
        """Get max timestamp to."""
        return get_min_time(get_current_time(), value="1t")

    def iter_all(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        step: str = "1d",
        retry: bool = False,
    ) -> Generator[Tuple[datetime, datetime], None, None]:
        """Iter all, by days in 1 hour chunks, further chunked by 1m intervals.

        1 day -> 24 hours -> 60 minutes or 10 minutes, etc.
        """
        for daily_timestamp_from, daily_timestamp_to, daily_existing in self.iter_days(
            timestamp_from, timestamp_to, step=step, retry=retry
        ):
            for hourly_timestamp_from, hourly_timestamp_to in self.iter_hours(
                daily_timestamp_from,
                daily_timestamp_to,
                daily_existing,
                retry=retry,
            ):
                yield hourly_timestamp_from, hourly_timestamp_to

    def iter_days(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        step: str = "1d",
        retry: bool = False,
    ):
        """Iter days."""
        for daily_timestamp_from, daily_timestamp_to in iter_timeframe(
            timestamp_from, timestamp_to, step, reverse=self.reverse
        ):
            # Query for daily.
            daily_existing = self.get_existing(
                daily_timestamp_from, daily_timestamp_to, retry=retry
            )
            daily_delta = daily_timestamp_to - daily_timestamp_from
            daily_expected = int(daily_delta.total_seconds() / 60)
            if len(daily_existing) < daily_expected:
                yield daily_timestamp_from, daily_timestamp_to, daily_existing

    def has_all_timestamps(
        self, timestamp_from: datetime, timestamp_to: datetime, existing: List[datetime]
    ) -> bool:
        """Has all timestamps."""
        delta = timestamp_to - timestamp_from
        expected = int(delta.total_seconds() / 60)
        return len(existing) == expected

    def iter_hours(
        self,
        daily_timestamp_from: datetime,
        daily_timestamp_to: datetime,
        daily_existing: List[datetime],
        reverse: bool = True,
        retry: bool = False,
    ):
        """Iter hours."""
        for timestamp_from, timestamp_to in iter_timeframe(
            daily_timestamp_from,
            daily_timestamp_to,
            value="1h",
            reverse=reverse,
        ):
            # List comprehension for hourly.
            existing = [
                timestamp
                for timestamp in daily_existing
                if timestamp >= timestamp_from and timestamp < timestamp_to
            ]
            if not self.has_all_timestamps(timestamp_from, timestamp_to, existing):
                for start_time, end_time in iter_missing(
                    timestamp_from,
                    timestamp_to,
                    existing,
                    reverse=reverse,
                ):
                    max_timestamp_to = self.get_max_timestamp_to()
                    end = max_timestamp_to if end_time > max_timestamp_to else end_time
                    if start_time != end:
                        if self.can_process(start_time, end):
                            yield start_time, end_time

    def can_process(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can process."""
        return True


class TradeDataIterator(BaseTimeFrameIterator):
    def __init__(self, symbol: Symbol) -> None:
        self.symbol = symbol
        # Trade data iterates from present to past.
        self.reverse = True

    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> List[datetime]:
        """Get existing."""
        queryset = TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )
        if retry:
            queryset = queryset.exclude(ok=False)
        return get_existing(queryset.values("timestamp", "frequency"))


class CandleCacheIterator(BaseTimeFrameIterator):
    def __init__(self, candle: Candle) -> None:
        self.candle = candle
        # Candle data iterates from past to present.
        self.reverse = False

    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> List[datetime]:
        """Get existing."""
        queryset = CandleCache.objects.filter(
            candle=self.candle,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )
        if retry:
            queryset = queryset.exclude(ok=False)
        return get_existing(queryset.values("timestamp", "frequency"))

    def can_process(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can process."""
        return all(
            [
                self.has_all_timestamps(
                    TradeData.get_existing(symbol), timestamp_from, timestamp_to
                )
                for symbol in self.candle.symbols.all()
            ]
        )
