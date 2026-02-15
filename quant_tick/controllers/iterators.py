from collections.abc import Generator
from datetime import datetime

from quant_tick.constants import Frequency
from quant_tick.lib import (
    get_current_time,
    get_existing,
    get_min_time,
    has_timestamps,
    iter_missing,
    iter_timeframe,
)
from quant_tick.models import Symbol, TradeData


class TradeDataIterator:
    """Trade data iterator for fetching from APIs."""

    def __init__(self, symbol: Symbol) -> None:
        """Initialize."""
        self.symbol = symbol

    def get_max_timestamp_to(self) -> datetime:
        """Get max timestamp to."""
        return get_min_time(get_current_time(), value="1min")

    def iter_all(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> Generator[tuple[datetime, datetime], None, None]:
        """Iter all, by days in 1 hour chunks, further chunked by 1m intervals.

        1 day -> 24 hours -> 60 minutes, etc.
        """
        for ts_from, ts_to, existing in self.iter_days(
            timestamp_from, timestamp_to, retry=retry
        ):
            yield from self.iter_hours(ts_from, ts_to, existing)

    def iter_days(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        retry: bool = False,
    ) -> Generator[tuple[datetime, datetime, list[datetime]], None, None]:
        """Iter days."""
        # Trade data iterates from present to past.
        for ts_from, ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=True
        ):
            existing = self.get_existing(ts_from, ts_to, retry=retry)
            if not has_timestamps(ts_from, ts_to, existing):
                yield ts_from, ts_to, existing

    def iter_hours(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        partition_existing: list[datetime],
    ) -> Generator[tuple[datetime, datetime], None, None]:
        """Iter hours."""
        for ts_from, ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1h", reverse=True
        ):
            existing = [
                timestamp
                for timestamp in partition_existing
                if timestamp >= ts_from and timestamp < ts_to
            ]
            if not has_timestamps(ts_from, ts_to, existing):
                for start_time, end_time in iter_missing(
                    ts_from, ts_to, existing, reverse=True
                ):
                    max_ts_to = self.get_max_timestamp_to()
                    end = max_ts_to if end_time > max_ts_to else end_time
                    if start_time != end:
                        yield start_time, end

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
