import logging
from datetime import datetime
from typing import Generator, List, Tuple

import pandas as pd
from django.conf import settings
from django.db import models
from django.db.models import Q

from quant_candles.constants import Frequency
from quant_candles.lib import (
    get_current_time,
    get_existing,
    get_min_time,
    has_timestamps,
    iter_missing,
    iter_timeframe,
)
from quant_candles.models import (
    Candle,
    CandleCache,
    CandleData,
    CandleReadOnlyData,
    Symbol,
    TimeBasedCandle,
    TradeData,
    TradeDataSummary,
)

logger = logging.getLogger(__name__)


def aggregate_trade_summary(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    retry: bool = False,
):
    """Aggregate trade summary."""
    for ts_from, ts_to in iter_timeframe(
        timestamp_from, timestamp_to, value="1d", reverse=True
    ):
        date = ts_from.date()
        has_trade_data_summary = (
            TradeDataSummary.objects.filter(symbol=symbol, date=date).exists()
            if not retry
            else False
        )
        has_trade_data = TradeData.objects.has_timestamps(symbol, ts_from, ts_to)
        if not has_trade_data_summary and has_trade_data:
            TradeDataSummary.aggregate(symbol, date)
            logger.info(
                "{symbol} summary: {date}".format(
                    **{"symbol": str(symbol), "date": date}
                )
            )


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
        retry: bool = False,
    ) -> Generator[Tuple[datetime, datetime], None, None]:
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
            if self.can_iter_hours():
                for hourly_timestamp_from, hourly_timestamp_to in self.iter_hours(
                    ts_from, ts_to, existing
                ):
                    yield hourly_timestamp_from, hourly_timestamp_to
            else:
                yield timestamp_from, timestamp_to

    def iter_days(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ):
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
        partition_existing: List[datetime],
    ):
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
    ) -> List[datetime]:
        """Get existing."""
        raise NotImplementedError

    def can_iter_days(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can iter range."""
        return True

    def can_iter_hours(self) -> bool:
        """Can iter hours."""
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


class TradeDataSummaryIterator(TradeDataIterator):
    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> List[datetime]:
        """Get existing."""
        queryset = TradeDataSummary.objects.filter(
            symbol=self.symbol, date=timestamp_from.date()
        )
        if retry:
            queryset = queryset.exclude(ok=False)
        return get_existing(queryset.values("timestamp", "frequency"))

    def can_iter_hours(self) -> bool:
        """Can iter hours."""
        return False

    def can_iter_days(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can iter days."""
        trade_data = (
            TradeData.objects.filter(
                symbol=self.symbol,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
            )
            .only("timestamp", "frequency")
            .values("timestamp", "frequency")
        )
        existing = get_existing(trade_data)
        return has_timestamps(timestamp_from, timestamp_to, existing)


class CandleCacheIterator(BaseTimeFrameIterator):
    def __init__(self, candle: Candle) -> None:
        self.candle = candle
        # Candle data iterates from past to present.
        self.reverse = False

    def get_existing(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> List[datetime]:
        """Get existing.

        Retry only locally, as the SQLite database not modifiable in Docker image.
        """
        query = Q(timestamp__gte=timestamp_from) & Q(timestamp__lt=timestamp_to)
        candle_cache = CandleCache.objects.filter(Q(candle=self.candle) & query)
        if settings.IS_LOCAL and retry:
            candle_data = CandleData.objects.filter(Q(candle=self.candle) & query)
            candle_read_only_data = CandleReadOnlyData.objects.filter(
                Q(candle_id=self.candle.id) & query
            )
            for queryset in (candle_cache, candle_data, candle_read_only_data):
                queryset.delete()
            return []
        else:
            return get_existing(candle_cache.values("timestamp", "frequency"))

    def can_iter_days(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can iter days."""
        return self.candle.can_aggregate(timestamp_from, timestamp_to)

    def can_iter_hours(self) -> bool:
        """Can iter hours."""
        is_time_based_candle = isinstance(self.candle, TimeBasedCandle)
        if is_time_based_candle:
            window = self.candle.json_data["window"]
            total_minutes = pd.Timedelta(window).total_seconds() / 60
            if int(total_minutes) > Frequency.HOUR.value:
                return False
        return True
