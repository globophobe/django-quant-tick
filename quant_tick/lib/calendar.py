import re
from collections.abc import Generator
from datetime import UTC, date, datetime, time

import pandas as pd
from pandas import Timestamp


def parse_datetime(value: str, unit: str = "ns") -> datetime:
    """Parse datetime with pandas for nanosecond accuracy."""
    return pd.to_datetime(value, unit=unit).replace(tzinfo=UTC)


def to_pydatetime(timestamp: Timestamp) -> datetime:
    return timestamp.replace(nanosecond=0).to_pydatetime().replace(tzinfo=UTC)


def to_utc_datetime(value: object) -> datetime:
    """Normalize pandas/native timestamp values to UTC datetime."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return to_pydatetime(timestamp)


def get_current_time(tzinfo: str = UTC) -> datetime:
    return datetime.utcnow().replace(tzinfo=tzinfo)


def get_min_time(timestamp: datetime, value: str) -> datetime:
    """Floor a timestamp to the requested interval."""
    match = re.match(r"\d+(\w+)", value)
    step = match.group(1)
    ts = pd.to_datetime(timestamp).floor(f"1{step}")
    return to_pydatetime(ts)


def get_next_time(timestamp: datetime, value: str) -> datetime:
    return get_min_time(timestamp, value=value) + pd.Timedelta(value)


def get_previous_time(timestamp: datetime, value: str) -> datetime:
    return get_min_time(timestamp, value=value) - pd.Timedelta(value)


def has_timestamps(
    timestamp_from: datetime, timestamp_to: datetime, existing: list[datetime]
) -> bool:
    delta = timestamp_to - timestamp_from
    expected = int(delta.total_seconds() / 60)
    actual = len(
        {
            timestamp
            for timestamp in existing
            if timestamp_from <= timestamp < timestamp_to
        }
    )
    return actual == expected


def timestamp_to_inclusive(
    timestamp_from: datetime, timestamp_to: datetime, value: str = "1min"
) -> datetime:
    """Reduce timestamp_to by value, in case results are inclusive."""
    ts_to = timestamp_to - pd.Timedelta(value)
    if timestamp_from <= ts_to:
        return ts_to
    else:
        return timestamp_to


def parse_fixed_resolution_minutes(value: str | int | None) -> int:
    """Parse a fixed-length resolution into whole minutes."""
    if value is None:
        return 1
    if isinstance(value, int):
        minutes = value
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError("resolution must not be blank")
        if raw[-1].isalpha():
            amount = int(raw[:-1])
            unit = raw[-1]
            if unit == "m":
                minutes = amount
            elif unit in {"h", "H"}:
                minutes = amount * 60
            elif unit in {"d", "D"}:
                minutes = amount * 1440
            elif unit in {"w", "W"}:
                minutes = amount * 10080
            elif unit == "M":
                raise ValueError("month resolutions are not fixed-length")
            else:
                raise ValueError(f"unsupported resolution unit: {unit}")
        else:
            minutes = int(raw)
    if minutes <= 0:
        raise ValueError("resolution must be positive minutes")
    return minutes


def get_interval_offset(value: str | int) -> pd.DateOffset | pd.Timedelta:
    """Get a pandas offset for a candle interval."""
    if isinstance(value, int):
        return pd.Timedelta(f"{value}min")
    raw = str(value).strip()
    if not raw:
        raise ValueError("interval must not be blank")
    if raw[-1].isalpha():
        amount = int(raw[:-1])
        unit = raw[-1]
    else:
        amount = int(raw)
        unit = "m"
    if unit == "s":
        return pd.Timedelta(seconds=amount)
    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit in {"h", "H"}:
        return pd.Timedelta(hours=amount)
    if unit in {"d", "D"}:
        return pd.Timedelta(days=amount)
    if unit in {"w", "W"}:
        return pd.Timedelta(weeks=amount)
    if unit == "M":
        return pd.DateOffset(months=amount)
    raise ValueError(f"unsupported interval: {value}")


def get_interval_inclusive_end(
    timestamp_from: datetime,
    timestamp_to: datetime,
    value: str | int,
) -> datetime:
    """Reduce timestamp_to by one interval for inclusive candle APIs."""
    ts_to = (pd.Timestamp(timestamp_to) - get_interval_offset(value)).to_pydatetime()
    return ts_to if timestamp_from <= ts_to else timestamp_to


def get_complete_interval_end(timestamp_to: datetime, value: str | int) -> datetime:
    """Floor an end timestamp to the last complete fixed interval boundary."""
    minutes = parse_fixed_resolution_minutes(value)
    return to_pydatetime(pd.Timestamp(timestamp_to).floor(f"{minutes}min"))


def get_interval_limit(
    timestamp_from: datetime,
    timestamp_to: datetime,
    value: str | int,
    max_results: int,
) -> int:
    """Count how many interval buckets fit in the requested range."""
    current = pd.Timestamp(timestamp_from)
    end = pd.Timestamp(timestamp_to)
    offset = get_interval_offset(value)
    limit = 0
    while current <= end and limit < max_results:
        limit += 1
        current = current + offset
    return limit


def parse_period_from_to(
    date_from: str | None = None,
    time_from: str | None = None,
    date_to: str | None = None,
    time_to: str | None = None,
) -> tuple[datetime]:
    """Parse period from/to command line arguments."""
    now = get_current_time()
    today = now.date()
    tomorrow = today + pd.Timedelta("1d")
    # timestamp_from
    date_from = date.fromisoformat(date_from) if date_from else date(2009, 1, 3)
    time_from = time.fromisoformat(time_from) if time_from else time.min
    # timestamp_to
    if date_to:
        date_to = date.fromisoformat(date_to)
    elif time_to:
        date_to = now.date()
    else:
        date_to = tomorrow
    time_to = time.fromisoformat(time_to) if time_to else time.min
    # UTC, please.
    timestamp_from = datetime.combine(date_from, time_from).replace(tzinfo=UTC)
    timestamp_to = datetime.combine(date_to, time_to).replace(tzinfo=UTC)
    # Sane defaults.
    timestamp_to = get_min_time(now, "1min") if timestamp_to >= now else timestamp_to
    timestamp_from = timestamp_to if timestamp_from > timestamp_to else timestamp_from
    return timestamp_from, timestamp_to


def get_range(
    timestamp_from: datetime, timestamp_to: datetime, value: str = "1min"
) -> list[datetime]:
    """Get timestamps in range, step by value."""
    ts_from_rounded = get_min_time(timestamp_from, value)
    ts_to_rounded = get_next_time(timestamp_to, value)
    return [
        to_pydatetime(timestamp)
        for timestamp in pd.date_range(ts_from_rounded, ts_to_rounded, freq=value)
        if timestamp >= ts_from_rounded and timestamp <= timestamp_to
    ]


def get_existing(values: list) -> list[datetime]:
    """Expand stored timestamp/frequency rows into minute timestamps."""
    result = []  # List of 1m timestamps.
    for item in values:
        timestamp = item["timestamp"]
        frequency = item["frequency"]
        result += [
            timestamp + pd.Timedelta(f"{index}min") for index in range(frequency)
        ]
    return sorted(result)


def get_missing(
    timestamp_from: datetime, timestamp_to: datetime, existing: list[datetime]
) -> list[datetime]:
    existing = set(existing)
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
    value: str = "1min",
    reverse: bool = False,
) -> Generator[tuple[datetime, datetime], None, None]:
    """Iterate adjacent timestamp windows at the requested step."""
    values = get_range(timestamp_from, timestamp_to, value)
    return iter_timestamps(values, reverse=reverse)


def iter_chunks(
    timestamp_from: datetime,
    timestamp_to: datetime,
    value: str | pd.Timedelta = "1d",
    reverse: bool = False,
) -> Generator[tuple[datetime, datetime], None, None]:
    """Iterate exact-size time chunks without rounding to interval boundaries."""
    step = pd.Timedelta(value)
    if step <= pd.Timedelta(0):
        raise ValueError("window value must be positive")

    start = pd.Timestamp(timestamp_from)
    end = pd.Timestamp(timestamp_to)
    if end <= start:
        return

    if reverse:
        cursor = end
        while start < cursor:
            next_cursor = max(start, cursor - step)
            yield to_pydatetime(next_cursor), to_pydatetime(cursor)
            cursor = next_cursor
    else:
        cursor = start
        while cursor < end:
            next_cursor = min(end, cursor + step)
            yield to_pydatetime(cursor), to_pydatetime(next_cursor)
            cursor = next_cursor


def iter_timestamps(
    values: list[datetime], reverse: bool = False
) -> Generator[tuple[datetime, datetime], None, None]:
    """Iter tuples of timestamps, optionally reversed."""
    if reverse:
        values.reverse()
    for start_time, end_time in zip(values, values[1:], strict=False):
        if reverse:
            yield end_time, start_time
        else:
            yield start_time, end_time


def iter_once(
    timestamp_from: datetime, timestamp_to: datetime
) -> Generator[tuple[datetime, datetime], None, None]:
    yield get_min_time(timestamp_from, "1d"), get_next_time(timestamp_to, "1d")


def iter_timeframe(
    timestamp_from: datetime,
    timestamp_to: datetime,
    value: str = "1d",
    reverse: bool = False,
) -> Generator[tuple[datetime, datetime], None, None]:
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
    existing: list[datetime],
    reverse: bool = False,
    value: str | pd.Timedelta = "1min",
    timestamps: list[datetime] | None = None,
) -> Generator[tuple[datetime, datetime], None, None]:
    """Iter missing windows for the requested interval."""
    step = pd.Timedelta(value)
    existing = set(existing)
    if timestamps is None:
        if isinstance(value, str):
            timestamps = get_range(timestamp_from, timestamp_to, value)
        else:
            timestamps = [
                to_pydatetime(timestamp)
                for timestamp in pd.date_range(timestamp_from, timestamp_to, freq=step)
            ]
    timestamps = sorted(
        timestamp
        for timestamp in timestamps
        if timestamp_from <= timestamp < timestamp_to
    )
    values = [
        (
            timestamp,
            min(timestamp_to, to_pydatetime(pd.Timestamp(timestamp) + step)),
        )
        for timestamp in timestamps
        if timestamp not in existing
    ]
    merged = []
    for ts_from, ts_to in values:
        if merged and merged[-1][1] == ts_from:
            merged[-1] = merged[-1][0], ts_to
        else:
            merged.append((ts_from, ts_to))
    if reverse:
        merged.reverse()
    return merged
