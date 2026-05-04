from datetime import UTC, datetime

import pandas as pd
from pandas import DataFrame


class ExchangeFunding:
    """Normalize exchange funding rows after adapter-specific parsing."""

    interval: pd.Timedelta | None = None
    anchor_offset = pd.Timedelta(0)
    timestamp_anomaly_tolerance = pd.Timedelta(0)

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
        from_ts = pd.to_datetime(timestamp_from, utc=True)
        to_ts = pd.to_datetime(timestamp_to, utc=True)
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
        df["_raw_timestamp_sort"] = raw_timestamps
        df = (
            df.sort_values(["timestamp", "_raw_timestamp_sort"], kind="stable")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .loc[lambda frame: (frame["timestamp"] >= from_ts) & (frame["timestamp"] < to_ts)]
            .drop(columns=["_raw_timestamp_sort"])
        )
        return df.set_index("timestamp")

    @classmethod
    def normalize_timestamp(cls, raw_timestamp: pd.Timestamp) -> tuple[datetime, int | None]:
        raw_timestamp = raw_timestamp.tz_convert(UTC)
        if cls.interval is None:
            timestamp = raw_timestamp.to_pydatetime()
            return timestamp, None

        anchor = pd.Timestamp("1970-01-01T00:00:00Z") + cls.anchor_offset
        interval_ns = cls.interval.value
        offset_ns = raw_timestamp.value - anchor.value
        bucket_ns = anchor.value + ((offset_ns + interval_ns // 2) // interval_ns) * interval_ns
        timestamp = pd.Timestamp(bucket_ns, tz=UTC)
        offset_ms = round((raw_timestamp - timestamp).total_seconds() * 1000)
        if offset_ms == 0:
            return timestamp.to_pydatetime(), None
        return timestamp.to_pydatetime(), offset_ms
