from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import filter_by_timestamp, iter_missing


class ExchangeFunding:
    interval: pd.Timedelta | None = None
    anchor_offset = pd.Timedelta(0)
    timestamp_anomaly_tolerance = pd.Timedelta(0)

    @classmethod
    def expected_timestamps(
        cls,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> list[datetime]:
        """Return expected normalized funding timestamps in a half-open range."""
        if cls.interval is None:
            return []

        from_ts = pd.to_datetime(timestamp_from, utc=True)
        to_ts = pd.to_datetime(timestamp_to, utc=True)
        if to_ts <= from_ts:
            return []

        anchor = cls.get_anchor()
        interval_ns = cls.interval.value
        start_step = -((anchor.value - from_ts.value) // interval_ns)
        timestamp = anchor + start_step * cls.interval
        timestamps = []
        while timestamp < to_ts:
            dt = timestamp.to_pydatetime()
            if not cls.is_known_missing_timestamp(dt):
                timestamps.append(dt)
            timestamp += cls.interval
        return timestamps

    @classmethod
    def missing_windows(
        cls,
        timestamp_from: datetime,
        timestamp_to: datetime,
        existing: set[datetime],
    ) -> list[tuple[datetime, datetime]]:
        if cls.interval is None:
            return []
        return iter_missing(
            timestamp_from,
            timestamp_to,
            existing,
            reverse=True,
            value=cls.interval,
            timestamps=cls.expected_timestamps(timestamp_from, timestamp_to),
        )

    @classmethod
    def get_anchor(cls) -> pd.Timestamp:
        return pd.Timestamp("1970-01-01T00:00:00Z") + cls.anchor_offset

    @classmethod
    def is_known_missing_timestamp(cls, timestamp: datetime) -> bool:
        return False

    @classmethod
    def empty_frame(cls, columns: list[str] | tuple[str, ...]) -> DataFrame:
        return DataFrame(columns=["timestamp", *columns]).set_index("timestamp")

    @classmethod
    def normalize_frame(
        cls,
        df: DataFrame,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> DataFrame:
        if df.empty:
            return df.set_index("timestamp")

        df = df.copy()
        raw_timestamps = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        timestamps = []
        raw_metadata = []
        offset_metadata = []
        anomaly_metadata = []
        for raw_ts in raw_timestamps:
            timestamp, offset = cls.normalize_timestamp(raw_ts)
            timestamps.append(timestamp)
            if offset:
                raw_metadata.append(raw_ts.to_pydatetime())
                offset_metadata.append(offset)
                anomaly_metadata.append(
                    abs(offset) > int(cls.timestamp_anomaly_tolerance.total_seconds() * 1000)
                )
            else:
                raw_metadata.append(None)
                offset_metadata.append(None)
                anomaly_metadata.append(None)

        df["timestamp"] = timestamps
        df["raw_timestamp"] = pd.Series(raw_metadata, dtype=object)
        df["timestamp_offset_ms"] = pd.Series(offset_metadata, dtype=object)
        df["timestamp_anomaly"] = pd.Series(anomaly_metadata, dtype=object)
        df = filter_by_timestamp(df, timestamp_from, timestamp_to)
        return df.set_index("timestamp")

    @classmethod
    def normalize_timestamp(cls, raw_timestamp: pd.Timestamp) -> tuple[datetime, int | None]:
        raw_timestamp = raw_timestamp.tz_convert(UTC)
        if cls.interval is None:
            timestamp = raw_timestamp.to_pydatetime()
            return timestamp, None

        anchor = cls.get_anchor()
        interval_ns = cls.interval.value
        offset_ns = raw_timestamp.value - anchor.value
        bucket_ns = anchor.value + ((offset_ns + interval_ns // 2) // interval_ns) * interval_ns
        timestamp = pd.Timestamp(bucket_ns, tz=UTC)
        offset_ms = round((raw_timestamp - timestamp).total_seconds() * 1000)
        if offset_ms == 0:
            return timestamp.to_pydatetime(), None
        return timestamp.to_pydatetime(), offset_ms
