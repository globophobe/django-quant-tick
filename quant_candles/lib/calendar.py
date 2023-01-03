from datetime import date, datetime, time, timezone
from typing import Generator, List, Optional, Tuple

import pandas as pd
from pandas import Timestamp


def parse_datetime(value: str, unit: str = "ns") -> datetime:
    """Parse datetime with pandas for nanosecond accuracy."""
    return pd.to_datetime(value, unit=unit).replace(tzinfo=timezone.utc)


def to_pydatetime(timestamp: Timestamp) -> datetime:
    """Timestamp to datetime."""
    return timestamp.replace(nanosecond=0).to_pydatetime().replace(tzinfo=timezone.utc)


def get_current_time(tzinfo=timezone.utc):
    """Get current time."""
    return datetime.utcnow().replace(tzinfo=tzinfo)


def get_min_time(timestamp: datetime, value: str) -> datetime:
    """Get minimum time."""
    step = value[-1]  # TODO: Refactor this.
    ts = pd.to_datetime(timestamp).floor(f"1{step}")
    return to_pydatetime(ts)


def get_next_time(timestamp: datetime, value: str) -> datetime:
    """Get next time."""
    return get_min_time(timestamp, value=value) + pd.Timedelta(value)


def get_previous_time(timestamp: datetime, value: str) -> datetime:
    """Get previous time."""
    return get_min_time(timestamp, value=value) - pd.Timedelta(value)


def get_next_monday(timestamp: datetime) -> datetime:
    """Get next Monday."""
    ts = get_min_time(timestamp, value="1d")
    weekday = ts.date().weekday()
    if weekday != 0:
        days = 7 - weekday % 7
        return ts + pd.Timedelta(f"{days}d")
    else:
        return ts + pd.Timedelta("7d")


def has_timestamps(
    timestamp_from: datetime, timestamp_to: datetime, existing: List[datetime]
) -> bool:
    """Has timestamps?"""
    delta = timestamp_to - timestamp_from
    expected = int(delta.total_seconds() / 60)
    return len(existing) == expected


def timestamp_to_inclusive(
    timestamp_from: datetime, timestamp_to: datetime, value: str = "1t"
):
    """Reduce timestamp_to by value, in case results are inclusive."""
    ts_to = timestamp_to - pd.Timedelta(value)
    if timestamp_from <= ts_to:
        return ts_to
    else:
        return timestamp_to


def parse_period_from_to(
    date_from: Optional[str] = None,
    time_from: Optional[str] = None,
    date_to: Optional[str] = None,
    time_to: Optional[str] = None,
) -> Tuple[datetime]:
    """Parse period from/to command line arguments."""
    now = get_current_time()
    tomorrow = now.date() + pd.Timedelta("1d")
    # timestamp_from
    date_from = date.fromisoformat(date_from) if date_from else date(2009, 1, 3)
    time_from = time.fromisoformat(time_from) if time_from else time.min
    # timestamp_to
    date_to = date.fromisoformat(date_to) if date_to else tomorrow
    time_to = time.fromisoformat(time_to) if time_to else time.min
    # UTC, please.
    timestamp_from = datetime.combine(date_from, time_from).replace(tzinfo=timezone.utc)
    timestamp_to = datetime.combine(date_to, time_to).replace(tzinfo=timezone.utc)
    # Sane defaults.
    timestamp_to = get_min_time(now, "1t") if timestamp_to >= now else timestamp_to
    timestamp_from = timestamp_to if timestamp_from > timestamp_to else timestamp_from
    return timestamp_from, timestamp_to


def get_range(timestamp_from: datetime, timestamp_to: datetime, value: str = "1t"):
    """Get timestamps in range, step by value."""
    ts_from_rounded = get_min_time(timestamp_from, value)
    ts_to_rounded = get_next_time(timestamp_to, value)
    return [
        to_pydatetime(timestamp)
        for timestamp in pd.date_range(ts_from_rounded, ts_to_rounded, freq=value)
        if timestamp >= ts_from_rounded and timestamp <= timestamp_to
    ]


def get_existing(values: List, retry: bool = False) -> List[datetime]:
    """Get existing."""
    result = []  # List of 1m timestamps.
    for item in values:
        timestamp = item["timestamp"]
        frequency = item["frequency"]
        result += [timestamp + pd.Timedelta(index) for index in range(frequency)]
    return sorted(result)


def get_missing(
    timestamp_from: datetime, timestamp_to: datetime, existing: List[datetime]
) -> List[datetime]:
    """Get missing."""
    return [
        timestamp
        for timestamp in get_range(timestamp_from, timestamp_to)
        if timestamp not in existing
        # Get missing result is assumed to not be inclusive.
        # However, result from get_range is.
        and timestamp != timestamp_to
    ]


def iter_window(
    timestamp_from: datetime,
    timestamp_to: datetime,
    value: str = "1t",
    reverse: bool = False,
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Iter window, by value."""
    values = get_range(timestamp_from, timestamp_to, value)
    return iter_timestamps(values, reverse=reverse)


def iter_timestamps(
    values: List[datetime], reverse: bool = False
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Iter tuples of timestamps, optionally reversed."""
    if reverse:
        values.reverse()
    for start_time, end_time in zip(values, values[1:]):
        if reverse:
            yield end_time, start_time
        else:
            yield start_time, end_time


def iter_once(
    timestamp_from: datetime, timestamp_to: datetime
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Fake iter, once."""
    yield get_min_time(timestamp_from, "1d"), get_next_time(timestamp_to, "1d")


def iter_timeframe(
    timestamp_from: datetime,
    timestamp_to: datetime,
    value: str = "1d",
    reverse: bool = False,
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Iter timeframe, including partial increments."""
    values = []
    head = None
    tail = None
    step = pd.Timedelta(value)
    # Is there at least 1 day?
    delta = timestamp_to - timestamp_from
    if delta >= step:
        ts_from = get_min_time(timestamp_from, value)
        ts_to = get_min_time(timestamp_to, value)
        # Is there a head?
        if timestamp_from != ts_from:
            head = timestamp_from, ts_from + step
            timestamp_from = ts_from + step
        # Is there a tail?
        if timestamp_to != ts_to:
            tail = ts_to, timestamp_to
            timestamp_to = ts_to
        # Check again, is there at least 1 day?
        delta = ts_to - ts_from
        if delta >= step:
            for val in iter_window(ts_from, ts_to, value, reverse=reverse):
                if head:
                    if val[0] >= head[0]:
                        values.append(val)
                elif tail:
                    if val[1] <= tail[1]:
                        values.append(val)
                else:
                    values.append(val)
    else:
        values.append((timestamp_from, timestamp_to))
    # Head or tail
    if head and not reverse:
        values.insert(0, head)
    elif tail and reverse:
        values.insert(0, tail)
    elif head and reverse:
        values.append(head)
    elif tail and not reverse:
        values.append(tail)
    for value in values:
        yield value


def iter_missing(
    timestamp_from: datetime,
    timestamp_to: datetime,
    existing: List[datetime],
    reverse: bool = False,
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Iter missing, by 1 minute intervals."""
    values = []
    for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, reverse=reverse):
        if ts_from not in existing:
            values.append((ts_from, ts_to))
    if reverse:
        values.reverse()
    if len(values):
        index = 0
        next_index = index + 1
        counter = 0
        one_minute = pd.Timedelta("1t")
        start = values[0][0]
        stop = None
        while next_index < len(values):
            next_start = values[next_index][0]
            total_minutes = one_minute * (counter + 1)
            if next_start == start + total_minutes:
                # Don't increment next_index, as value will be deleted.
                stop = values[next_index][1]
                del values[next_index]
                counter += 1
            else:
                if stop:
                    values[index] = start, stop
                    index = next_index
                    stop = None
                start = next_start
                counter = 0
                next_index += 1
        if stop:
            values[-1] = values[-1][0], stop
    if reverse:
        values.reverse()
    return values
