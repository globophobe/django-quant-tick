import datetime
import logging
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    calculate_notional,
    calculate_tick_rule,
    filter_by_timestamp,
    get_current_time,
    get_min_time,
    gzip_downloader,
    set_dtypes,
)

from .base import BaseController
from .iterators import TradeDataIterator

logger = logging.getLogger(__name__)


def use_s3() -> datetime:
    date = get_current_time().date() - pd.Timedelta("2d")
    return datetime.datetime.combine(date, datetime.time.min).replace(
        tzinfo=datetime.UTC
    )


class ExchangeS3(BaseController):
    """Base controller for daily exchange S3 archives."""

    archive_publication_grace_days = 7
    missing_archive_dates = frozenset()

    def get_url(self, date: datetime.date) -> str:
        raise NotImplementedError

    @property
    def gzipped_csv_columns(self) -> list:
        return [
            "timestamp",
            "symbol",
            "side",
            "size",
            "price",
            "tickDirection",
            "trdMatchID",
            "grossValue",
            "foreignNotional",
        ]

    @property
    def columns(self) -> list:
        return [
            "uid",
            "timestamp",
            "nanoseconds",
            "price",
            "volume",
            "notional",
            "tickRule",
        ]

    def main(self) -> None:
        """Fetch daily S3 files and persist matching partitions."""
        iterator = TradeDataIterator(self.symbol)
        for timestamp_from, timestamp_to, existing in iterator.iter_days(
            self.timestamp_from,
            self.timestamp_to,
            retry=self.retry,
        ):
            date = timestamp_from.date()
            data_frame = self.get_data_frame(date)
            if data_frame is not None:
                is_full_day = (
                    timestamp_from.time() == datetime.time.min
                    and timestamp_to == timestamp_from + pd.Timedelta("1d")
                )
                if existing or not is_full_day:
                    windows = iterator.iter_hours(
                        timestamp_from, timestamp_to, existing
                    )
                else:
                    windows = ((timestamp_from, timestamp_to),)
                for ts_from, ts_to in windows:
                    df = filter_by_timestamp(data_frame, ts_from, ts_to)
                    candles = self.get_candles(ts_from, ts_to)
                    self.on_data_frame(self.symbol, ts_from, ts_to, df, candles)
            elif self._should_stop_after_missing_archive(date):
                break

    def _should_stop_after_missing_archive(self, date: datetime.date) -> bool:
        if date in self.missing_archive_dates:
            return False
        cutoff = get_current_time().date() - datetime.timedelta(
            days=self.archive_publication_grace_days
        )
        return date < cutoff

    def get_data_frame(self, date: datetime.date) -> DataFrame | None:
        """Download and parse one daily S3 file."""
        url = self.get_url(date)
        data_frame = gzip_downloader(url, self.gzipped_csv_columns)
        if data_frame is not None:
            df = self.filter_by_symbol(data_frame)
            if len(df):
                return self.parse_dtypes_and_strip_columns(df)
            return df

    def filter_by_symbol(self, data_frame: DataFrame) -> DataFrame:
        """Keep rows for the configured symbol."""
        if "symbol" in data_frame.columns:
            return data_frame[data_frame.symbol == self.symbol.api_symbol]
        else:
            return data_frame

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse S3 columns into the canonical trade schema."""
        data_frame = set_dtypes(data_frame)
        data_frame = calculate_notional(data_frame)
        data_frame = calculate_tick_rule(data_frame)
        return data_frame[self.columns]


class ChunkedExchangeS3(ExchangeS3):
    """Process complete daily archives in bounded timestamp frames."""

    archive_chunksize = 200_000
    allow_descending_archive = False

    def get_data_frame_chunks(
        self,
        value: datetime.date,
    ) -> Iterable[DataFrame] | None:
        raise NotImplementedError

    def prepare_archive_chunk(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    def get_archive_timestamp_nanoseconds(
        self,
        data_frame: DataFrame,
    ) -> pd.Series:
        raise NotImplementedError

    def _combine_archive_hour(
        self,
        parts: list[DataFrame],
        *,
        descending: bool,
    ) -> DataFrame:
        frame = (
            parts[0]
            if len(parts) == 1
            else pd.concat(parts, ignore_index=True, copy=False)
        )
        if descending and len(frame) > 1:
            timestamps = self.get_archive_timestamp_nanoseconds(frame)
            order = np.argsort(
                timestamps.to_numpy(dtype="int64", copy=False),
                kind="stable",
            )
            frame = frame.iloc[order].reset_index(drop=True)
        return frame

    def iter_archive_hours(
        self,
        data_frames: Iterable[DataFrame],
    ) -> Iterator[tuple[datetime.datetime, DataFrame]]:
        current_hour = None
        current_parts = []
        previous_timestamp = None
        direction = 0
        nanoseconds_per_hour = 3_600_000_000_000

        for data_frame in data_frames:
            data_frame = self.prepare_archive_chunk(data_frame)
            if not len(data_frame):
                continue
            timestamps = self.get_archive_timestamp_nanoseconds(data_frame)
            values = timestamps.to_numpy(dtype="int64", copy=False)
            increases = False
            decreases = False
            if previous_timestamp is not None:
                increases = values[0] > previous_timestamp
                decreases = values[0] < previous_timestamp
            if len(values) > 1:
                increases = increases or bool(np.any(values[1:] > values[:-1]))
                decreases = decreases or bool(np.any(values[1:] < values[:-1]))
            if increases and decreases:
                raise ValueError("archive timestamps are not monotonic")
            if decreases and not self.allow_descending_archive:
                raise ValueError("archive timestamps are not monotonic")
            chunk_direction = 1 if increases else -1 if decreases else 0
            if direction and chunk_direction and direction != chunk_direction:
                raise ValueError("archive timestamps are not monotonic")
            direction = direction or chunk_direction
            previous_timestamp = int(values[-1])

            hours = values // nanoseconds_per_hour
            boundaries = np.flatnonzero(hours[1:] != hours[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            stops = np.concatenate((boundaries, [len(data_frame)]))
            for start, stop in zip(starts, stops, strict=True):
                hour = int(hours[start])
                if current_hour is not None and hour != current_hour:
                    timestamp = pd.Timestamp(
                        current_hour * nanoseconds_per_hour,
                        unit="ns",
                        tz="UTC",
                    ).to_pydatetime()
                    frame = self._combine_archive_hour(
                        current_parts,
                        descending=direction < 0,
                    )
                    current_parts = []
                    yield timestamp, frame
                    del frame
                current_hour = hour
                current_parts.append(data_frame.iloc[start:stop])

        if current_hour is not None:
            timestamp = pd.Timestamp(
                current_hour * nanoseconds_per_hour,
                unit="ns",
                tz="UTC",
            ).to_pydatetime()
            frame = self._combine_archive_hour(
                current_parts,
                descending=direction < 0,
            )
            current_parts = []
            yield timestamp, frame

    def main(self) -> None:
        """Fetch daily archives and persist bounded hourly frames."""
        iterator = TradeDataIterator(self.symbol)
        for timestamp_from, timestamp_to, existing in iterator.iter_days(
            self.timestamp_from,
            self.timestamp_to,
            retry=self.retry,
        ):
            chunks = self.get_data_frame_chunks(timestamp_from.date())
            if chunks is None:
                if self._should_stop_after_missing_archive(timestamp_from.date()):
                    break
                continue
            windows = sorted(
                iterator.iter_hours(timestamp_from, timestamp_to, existing),
                key=lambda value: value[0],
            )
            windows_by_hour = {}
            for window_from, window_to in windows:
                hour = get_min_time(window_from, "1h")
                windows_by_hour.setdefault(hour, []).append((window_from, window_to))
            try:
                candles = self.get_candles(timestamp_from, timestamp_to)
                for hour, raw_data in self.iter_archive_hours(chunks):
                    hour_windows = windows_by_hour.pop(hour, ())
                    if not hour_windows:
                        del raw_data
                        continue
                    data_frame = self.parse_dtypes_and_strip_columns(raw_data)
                    for window_from, window_to in hour_windows:
                        self._write_archive_window(
                            window_from,
                            window_to,
                            data_frame,
                            candles,
                        )
                    del data_frame, raw_data
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()

            empty = pd.DataFrame(columns=self.columns)
            for hour_windows in windows_by_hour.values():
                for window_from, window_to in hour_windows:
                    self._write_archive_window(
                        window_from,
                        window_to,
                        empty,
                        candles,
                    )

    def _write_archive_window(
        self,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        data_frame: DataFrame,
        candles: DataFrame,
    ) -> None:
        trades = filter_by_timestamp(data_frame, timestamp_from, timestamp_to)
        window_candles = filter_by_timestamp(
            candles,
            timestamp_from,
            timestamp_to,
        )
        self.on_data_frame(
            self.symbol,
            timestamp_from,
            timestamp_to,
            trades,
            window_candles,
        )
