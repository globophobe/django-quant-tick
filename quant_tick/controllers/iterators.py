import logging
from collections.abc import Generator
from datetime import datetime

import pandas as pd
from django.db.models import Q

from quant_tick.constants import Frequency
from quant_tick.lib import (
    get_current_time,
    get_existing,
    get_min_time,
    has_timestamps,
    iter_missing,
    iter_timeframe,
)
from quant_tick.models import (
    Candle,
    CandleCache,
    CandleData,
    Symbol,
    TimeBasedCandle,
    TradeData,
)

logger = logging.getLogger(__name__)


def aggregate_candles(
    candle: Candle,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
) -> None:
    """Aggregate candles."""
    # First, adjust timestamps.
    min_timestamp_from, max_timestamp_to, cache_data = candle.initialize(
        timestamp_from, timestamp_to, retry
    )
    # Next, aggregate candles.
    for ts_from, ts_to in CandleCacheIterator(candle).iter_all(
        min_timestamp_from, max_timestamp_to, retry=retry
    ):
        data_frame = candle.get_data_frame(ts_from, ts_to)
        cache_data = candle.get_cache_data(ts_from, cache_data)
        data, cache_data = candle.aggregate(ts_from, ts_to, data_frame, cache_data)
        candle.write_cache(ts_from, ts_to, cache_data)
        candle.write_data(ts_from, ts_to, data)
        logger.info(
            "Candle {candle}: {timestamp}".format(
                **{"candle": str(candle), "timestamp": ts_to.replace(tzinfo=None)}
            )
        )


class BaseTimeFrameIterator:
    """Base time frame iterator."""

    def __init__(self) -> None:
        """Initialize."""
        self.reverse = None

    def get_max_timestamp_to(self) -> datetime:
        """Get max timestamp to."""
        return get_min_time(get_current_time(), value="1min")

    def iter_all(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> Generator[tuple[datetime, datetime], None, None]:
        """Iter all, default by days in 1 hour chunks, further chunked by 1m intervals.

        1 day -> 24 hours -> 60 minutes or 10 minutes, etc.

        Trade data is saved at intervals of 1 hour or 1 minute. Candles are saved at
        intervals ranging from 1 day to 1 minute. Step size greater than 1 day is better
        avoided, as it could mean opening more than 24 files in memory.

        Accordingly, aggregations with step greater than 1 day should use daily candles,
        and not trade data.
        """
        for ts_from, ts_to, existing in self.iter_days(
            timestamp_from, timestamp_to, retry=retry
        ):
            if self.can_iter_hours(ts_from, ts_to):
                yield from self.iter_hours(ts_from, ts_to, existing)
            else:
                yield timestamp_from, timestamp_to

    def iter_days(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> Generator[tuple[datetime, datetime], None, None]:
        """Iter days."""
        for ts_from, ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=self.reverse
        ):
            existing = self.get_existing(ts_from, ts_to, retry=retry)
            if not has_timestamps(ts_from, ts_to, existing):
                if self.can_iter_days(ts_from, ts_to):
                    yield ts_from, ts_to, existing

    def iter_hours(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        partition_existing: list[datetime],
    ) -> Generator[tuple[datetime, datetime], None, None]:
        """Iter hours."""
        for ts_from, ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1h", reverse=self.reverse
        ):
            # List comprehension for hourly.
            existing = [
                timestamp
                for timestamp in partition_existing
                if timestamp >= ts_from and timestamp < ts_to
            ]
            if not has_timestamps(ts_from, ts_to, existing):
                for start_time, end_time in iter_missing(
                    ts_from, ts_to, existing, reverse=self.reverse
                ):
                    max_timestamp_to = self.get_max_timestamp_to()
                    end = max_timestamp_to if end_time > max_timestamp_to else end_time
                    if start_time != end:
                        yield start_time, end_time

    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> list[datetime]:
        """Get existing."""
        raise NotImplementedError

    def can_iter_days(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can iter days."""
        return True

    def can_iter_hours(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can iter hours."""
        return True


class TradeDataIterator(BaseTimeFrameIterator):
    """Trade data iterator."""

    def __init__(self, symbol: Symbol) -> None:
        """Initialize."""
        self.symbol = symbol
        # Trade data iterates from present to past.
        self.reverse = True

    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> list[datetime]:
        """Get existing."""
        queryset = TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
        )
        if retry:
            # Delete daily data.
            queryset.filter(frequency=Frequency.DAY, ok=False).delete()
            # Overwrite hourly or minute data.
            queryset = queryset.exclude(ok=False)
        return get_existing(queryset.values("timestamp", "frequency"))


class CandleCacheIterator(BaseTimeFrameIterator):
    """Candle cache iterator."""

    def __init__(self, candle: Candle) -> None:
        """Initialize."""
        self.candle = candle
        # Candle data iterates from past to present.
        self.reverse = False

    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> list[datetime]:
        """Get existing."""
        query = (
            Q(candle=self.candle)
            & Q(timestamp__gte=timestamp_from)
            & Q(timestamp__lt=timestamp_to)
        )
        candle_cache = CandleCache.objects.filter(query)
        candle_data = CandleData.objects.filter(query)
        if retry:
            for queryset in (candle_cache, candle_data):
                queryset.delete()
            return []
        else:
            return get_existing(candle_cache.values("timestamp", "frequency"))

    def can_iter_days(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can iter days."""
        return self.candle.can_aggregate(timestamp_from, timestamp_to)

    def can_iter_hours(self, *args) -> bool:
        """Can iter hours."""
        is_time_based_candle = isinstance(self.candle, TimeBasedCandle)
        if is_time_based_candle:
            window = self.candle.json_data["window"]
            total_minutes = pd.Timedelta(window).total_seconds() / 60
            if int(total_minutes) > Frequency.HOUR:
                return False
        return True
