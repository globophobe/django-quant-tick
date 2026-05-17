import logging
import os
import time
from collections.abc import Callable
from datetime import datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    assert_type_decimal,
    filter_by_timestamp,
    get_current_time,
    get_min_time,
    iter_window,
)
from quant_tick.models import Symbol, TradeData, WebSocketData

from .base import BaseController
from .iterators import TradeDataIterator

logger = logging.getLogger(__name__)
WEBSOCKET_DATA_LOOKBACK = pd.Timedelta("30min")
# Default ingestion normally uses 10-minute partitions. For that path, more than one invalid
# websocket range is noisy enough that a full REST fetch is simpler. Longer/manual
# partitions may tolerate two invalid ranges before falling back to REST-only.
WEBSOCKET_REST_BACKFILL_SHORT_PARTITION_MAX_RANGES = 1
WEBSOCKET_REST_BACKFILL_LONG_PARTITION_MAX_RANGES = 2
WEBSOCKET_REST_BACKFILL_SHORT_PARTITION = pd.Timedelta("10min")


def iter_api(
    url: str,
    get_api_pagination_id: Callable,
    get_api_timestamp: Callable,
    get_api_response: Callable,
    max_results: int,
    min_elapsed_per_request: int,
    timestamp_from: datetime | None = None,
    pagination_id: str | None = None,
    log_format: str | None = None,
) -> list:
    """Iterate a paginated exchange API until the partition is covered."""
    results = []
    last_data = []
    stop_iteration = False
    while not stop_iteration:
        start = time.time()
        data = get_api_response(
            url, timestamp_from=timestamp_from, pagination_id=pagination_id
        )
        if not len(data):
            is_last_iteration = stop_iteration = True
        else:
            last_trade = data[-1]
            timestamp = get_api_timestamp(last_trade)
            # Next pagination_id
            pagination_id = get_api_pagination_id(
                timestamp, last_data=last_data, data=data
            )
            # B/C unique
            last_data = data
            # Append results
            results += data
            less_than_max_results = len(data) < max_results
            is_last_iteration = pagination_id is None or less_than_max_results
            is_within_partition = timestamp_from and timestamp > timestamp_from
            # Maybe stop iteration
            if is_last_iteration or not is_within_partition:
                stop_iteration = True
            # Basic logging for stackdriver
            if log_format and timestamp:
                t = timestamp.replace(tzinfo=None).isoformat()
                if not timestamp.microsecond:
                    t += ".000000"
                logger.info(log_format.format(**{"timestamp": t}))
        # Throttle requests
        elapsed = time.time() - start
        if elapsed < min_elapsed_per_request:
            time.sleep(min_elapsed_per_request - elapsed)
    return results, is_last_iteration, pagination_id


def get_api_max_requests_reset(seconds: int) -> float:
    return time.time() + seconds


def set_api_environ_vars(
    max_requests_reset_key: str,
    total_requests_key: str,
    max_requests_reset: float,
    reset: bool = False,
) -> None:
    if reset or (max_requests_reset_key not in os.environ):
        max_requests_reset = get_api_max_requests_reset(max_requests_reset)
        os.environ[max_requests_reset_key] = str(max_requests_reset)
    if reset or (total_requests_key not in os.environ):
        os.environ[total_requests_key] = str(0)


def increment_api_total_requests(total_requests_key: str) -> None:
    total_requests = int(os.environ[total_requests_key])
    os.environ[total_requests_key] = str(total_requests + 1)


def throttle_api_requests(
    max_requests_reset_key: str,
    total_requests_key: str,
    max_requests_reset: float,
    max_requests: int,
) -> None:
    """Throttle requests using env-backed request counters."""
    set_api_environ_vars(
        max_requests_reset_key,
        total_requests_key,
        max_requests_reset,
        reset=False,
    )
    now = time.time()
    value = float(os.environ[max_requests_reset_key])
    if now >= value:
        set_api_environ_vars(
            max_requests_reset_key,
            total_requests_key,
            max_requests_reset,
            reset=True,
        )
    else:
        total_requests = int(os.environ[total_requests_key])
        if total_requests >= max_requests:
            sleep_time = float(os.environ[max_requests_reset_key]) - now
            if sleep_time > 0:
                logger.info(f"Max requests, sleeping {sleep_time} seconds")
                time.sleep(sleep_time)


class ExchangeREST(BaseController):
    """Base controller for REST trade ingestion."""

    def get_pagination_id(self, timestamp_to: datetime) -> None:
        raise NotImplementedError

    def iter_api(self, symbol: Symbol, pagination_id: str, log_format: str) -> None:
        raise NotImplementedError

    def get_max_timestamp_to(self) -> datetime:
        """Get the latest safe timestamp_to for exchange data."""

    def get_timestamp_to(
        self, timestamp_to: datetime, max_timestamp_to: datetime
    ) -> datetime:
        """Clamp timestamp_to to complete exchange data."""

    def main(self) -> DataFrame:
        """Fetch, normalize, validate, and persist TradeData slices."""
        buffered_trades = []
        pagination_id = None
        is_last_iteration = False
        previous_timestamp_from = None
        for timestamp_from, timestamp_to in TradeDataIterator(self.symbol).iter_all(
            self.timestamp_from,
            self.timestamp_to,
            retry=self.retry,
        ):
            candles = self.get_candles(timestamp_from, timestamp_to)
            websocket_partitions = (
                {}
                if self.retry
                else self.validate_websocket_partitions(
                    timestamp_from,
                    timestamp_to,
                    candles,
                )
            )
            websocket_timestamp_from = max(
                timestamp_from,
                self.get_websocket_timestamp_from(),
            )
            has_zero_websocket_range = (
                not self.retry
                and websocket_timestamp_from < timestamp_to
                and self.has_zero_trade_candles(
                    candles,
                    websocket_timestamp_from,
                    timestamp_to,
                )
            )
            if websocket_partitions or has_zero_websocket_range:
                self.on_websocket_partitions(
                    timestamp_from,
                    timestamp_to,
                    candles,
                    websocket_partitions,
                )
                buffered_trades = []
                pagination_id = None
                is_last_iteration = False
                previous_timestamp_from = None
                continue
            if previous_timestamp_from == timestamp_to:
                buffered_trades = [
                    trade
                    for trade in buffered_trades
                    if trade["timestamp"] < timestamp_to
                ]
                if pagination_id is None:
                    pagination_id = self.get_pagination_id(timestamp_to)
            else:
                buffered_trades = []
                pagination_id = self.get_pagination_id(timestamp_to)
                is_last_iteration = False
            oldest_timestamp = self.get_oldest_timestamp(buffered_trades)
            while not is_last_iteration and (
                oldest_timestamp is None or oldest_timestamp > timestamp_from
            ):
                trade_data, is_last_iteration, pagination_id = self.iter_api(
                    timestamp_from,
                    pagination_id,
                )
                buffered_trades += self.parse_data(trade_data)
                oldest_timestamp = self.get_oldest_timestamp(buffered_trades)
            valid_trades, buffered_trades = self.split_trades(
                timestamp_from,
                timestamp_to,
                buffered_trades,
            )
            data_frame = self.get_data_frame(valid_trades)
            if len(data_frame):
                self.assert_data_frame(
                    timestamp_from,
                    timestamp_to,
                    data_frame,
                    valid_trades,
                )
            self.on_data_frame(
                self.symbol, timestamp_from, timestamp_to, data_frame, candles
            )
            previous_timestamp_from = timestamp_from
            # Complete
            if is_last_iteration and not buffered_trades:
                break

    def validate_websocket_partitions(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        candles: DataFrame,
    ) -> dict[datetime, tuple[DataFrame | None, DataFrame | None, DataFrame | None]]:
        timestamp_from = max(timestamp_from, self.get_websocket_timestamp_from())
        if timestamp_from >= timestamp_to:
            return {}

        rows = (
            WebSocketData.objects.for_symbol(self.symbol)
            .filter(timestamp__gte=timestamp_from, timestamp__lt=timestamp_to)
            .order_by("timestamp")
        )
        partitions = {}
        for row in rows:
            bucket_to = row.timestamp + pd.Timedelta("1min")
            if bucket_to > timestamp_to:
                continue
            frames = self.validate_websocket_partition(row, bucket_to, candles)
            if frames is not None:
                partitions[row.timestamp] = frames
        return partitions

    @staticmethod
    def get_websocket_timestamp_from() -> datetime:
        return get_min_time(get_current_time() - WEBSOCKET_DATA_LOOKBACK, "1min")

    def validate_websocket_partition(
        self,
        row: WebSocketData,
        timestamp_to: datetime,
        candles: DataFrame,
    ) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None] | None:
        raw_trades, aggregated_trades, filtered_trades = row.get_data_frames(
            self.symbol
        )
        if raw_trades is None and aggregated_trades is None and filtered_trades is None:
            return None
        try:
            ok = TradeData.validate(
                self.symbol,
                row.timestamp,
                timestamp_to,
                candles,
                raw_trades=raw_trades,
                aggregated_trades=aggregated_trades,
                filtered_trades=filtered_trades,
            )
            if ok is not True:
                return None
            raw_trades, aggregated_trades, filtered_trades = (
                TradeData._prepare_partition_data(
                    self.symbol,
                    row.timestamp,
                    timestamp_to,
                    raw_trades=raw_trades,
                    aggregated_trades=aggregated_trades,
                    filtered_trades=filtered_trades,
                )
            )
        except Exception:
            logger.exception("%s: websocket data validation failed", self.symbol)
            return None
        logger.info(
            "%s: websocket data validation ok=%s timestamp=%s",
            self.symbol,
            ok,
            row.timestamp.isoformat(),
        )
        return self.get_enabled_trade_frames(
            raw_trades,
            aggregated_trades,
            filtered_trades,
        )

    def on_websocket_partitions(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        candles: DataFrame,
        websocket_partitions: dict[
            datetime, tuple[DataFrame | None, DataFrame | None, DataFrame | None]
        ],
    ) -> None:
        raw_frames: list[DataFrame] = []
        aggregated_frames: list[DataFrame] = []
        filtered_frames: list[DataFrame] = []

        def append_frames(
            raw_trades: DataFrame | None,
            aggregated_trades: DataFrame | None,
            filtered_trades: DataFrame | None,
        ) -> None:
            if raw_trades is not None:
                raw_frames.append(raw_trades)
            if aggregated_trades is not None:
                aggregated_frames.append(aggregated_trades)
            if filtered_trades is not None:
                filtered_frames.append(filtered_trades)

        def append_rest_range(
            ts_from: datetime,
            ts_to: datetime,
            data_frame: DataFrame | None = None,
        ) -> None:
            data_frame = (
                self.fetch_rest_data_frame(ts_from, ts_to)
                if data_frame is None
                else data_frame
            )
            raw_trades, aggregated_trades, filtered_trades = (
                TradeData._prepare_partition_data(
                    self.symbol,
                    ts_from,
                    ts_to,
                    raw_trades=data_frame,
                )
            )
            append_frames(
                *self.get_enabled_trade_frames(
                    raw_trades,
                    aggregated_trades,
                    filtered_trades,
                )
            )

        websocket_timestamp_from = max(
            timestamp_from,
            self.get_websocket_timestamp_from(),
        )
        invalid_ranges = self.get_invalid_websocket_ranges(
            websocket_timestamp_from,
            timestamp_to,
            websocket_partitions,
            candles,
        )
        if self.should_fetch_rest_only_for_websocket_ranges(
            websocket_timestamp_from,
            timestamp_to,
            invalid_ranges,
        ):
            self.on_rest_data_frame(timestamp_from, timestamp_to, candles)
            return

        if timestamp_from < websocket_timestamp_from:
            append_rest_range(timestamp_from, websocket_timestamp_from)

        rest_data_frame = None
        if invalid_ranges:
            # Fetch one REST span that covers all invalid ranges, then slice each
            # missing minute from that frame below. This avoids one REST request per
            # bad minute while still preserving valid websocket minutes.
            rest_data_frame = self.fetch_rest_data_frame(
                invalid_ranges[0][0],
                invalid_ranges[-1][1],
            )

        for ts_from, ts_to in iter_window(
            websocket_timestamp_from,
            timestamp_to,
            value="1min",
        ):
            frames = websocket_partitions.get(ts_from)
            if frames is None:
                # Match REST validation semantics: an empty trade minute is valid
                # when the exchange candle has zero volume/notional.
                if self.has_zero_trade_candle(candles, ts_from, ts_to):
                    continue
                append_rest_range(ts_from, ts_to, rest_data_frame)
                continue
            append_frames(*frames)

        raw_trades = self.concat_trade_frames(raw_frames)
        aggregated_trades = self.concat_trade_frames(aggregated_frames)
        filtered_trades = self.concat_trade_frames(filtered_frames)
        data_frame = TradeData._get_validation_frame(
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        if data_frame is None:
            data_frame = pd.DataFrame([])
        self.on_data_frame(
            self.symbol,
            timestamp_from,
            timestamp_to,
            data_frame,
            candles,
            **self.get_data_frame_kwargs(
                raw_trades,
                aggregated_trades,
                filtered_trades,
            ),
        )

    def get_invalid_websocket_ranges(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        websocket_partitions: dict[
            datetime, tuple[DataFrame | None, DataFrame | None, DataFrame | None]
        ],
        candles: DataFrame,
    ) -> list[tuple[datetime, datetime]]:
        ranges = []
        range_from = None
        range_to = None
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, value="1min"):
            if ts_from in websocket_partitions or self.has_zero_trade_candle(
                candles,
                ts_from,
                ts_to,
            ):
                if range_from is not None:
                    ranges.append((range_from, range_to))
                    range_from = None
                    range_to = None
                continue
            if range_from is None:
                range_from = ts_from
            range_to = ts_to
        if range_from is not None:
            ranges.append((range_from, range_to))
        return ranges

    @staticmethod
    def has_zero_trade_candle(
        candles: DataFrame,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> bool:
        if not len(candles):
            return False
        if "notional" in candles.columns:
            key = "notional"
        elif "volume" in candles.columns:
            key = "volume"
        else:
            return False
        candle = filter_by_timestamp(candles, timestamp_from, timestamp_to)
        return len(candle) > 0 and candle[key].sum() == 0

    @classmethod
    def has_zero_trade_candles(
        cls,
        candles: DataFrame,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> bool:
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, value="1min"):
            if not cls.has_zero_trade_candle(candles, ts_from, ts_to):
                return False
        return True

    @staticmethod
    def should_fetch_rest_only_for_websocket_ranges(
        timestamp_from: datetime,
        timestamp_to: datetime,
        invalid_ranges: list[tuple[datetime, datetime]],
    ) -> bool:
        max_ranges = (
            WEBSOCKET_REST_BACKFILL_SHORT_PARTITION_MAX_RANGES
            if timestamp_to - timestamp_from <= WEBSOCKET_REST_BACKFILL_SHORT_PARTITION
            else WEBSOCKET_REST_BACKFILL_LONG_PARTITION_MAX_RANGES
        )
        return len(invalid_ranges) > max_ranges

    def on_rest_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        candles: DataFrame,
    ) -> None:
        data_frame = self.fetch_rest_data_frame(timestamp_from, timestamp_to)
        self.on_data_frame(
            self.symbol,
            timestamp_from,
            timestamp_to,
            data_frame,
            candles,
        )

    def fetch_rest_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> DataFrame:
        buffered_trades = []
        pagination_id = self.get_pagination_id(timestamp_to)
        is_last_iteration = False
        oldest_timestamp = self.get_oldest_timestamp(buffered_trades)
        while not is_last_iteration and (
            oldest_timestamp is None or oldest_timestamp > timestamp_from
        ):
            trade_data, is_last_iteration, pagination_id = self.iter_api(
                timestamp_from,
                pagination_id,
            )
            buffered_trades += self.parse_data(trade_data)
            oldest_timestamp = self.get_oldest_timestamp(buffered_trades)
        valid_trades, _ = self.split_trades(
            timestamp_from,
            timestamp_to,
            buffered_trades,
        )
        data_frame = self.get_data_frame(valid_trades)
        if len(data_frame):
            self.assert_data_frame(
                timestamp_from,
                timestamp_to,
                data_frame,
                valid_trades,
            )
        return data_frame

    def get_enabled_trade_frames(
        self,
        raw_trades: DataFrame | None,
        aggregated_trades: DataFrame | None,
        filtered_trades: DataFrame | None,
    ) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None]:
        if not self.symbol.save_raw:
            raw_trades = None
        if not self.symbol.save_aggregated:
            aggregated_trades = None
        if not self.symbol.significant_trade_filter:
            filtered_trades = None
        return raw_trades, aggregated_trades, filtered_trades

    @staticmethod
    def concat_trade_frames(frames: list[DataFrame]) -> DataFrame | None:
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def get_data_frame_kwargs(
        raw_trades: DataFrame | None,
        aggregated_trades: DataFrame | None,
        filtered_trades: DataFrame | None,
    ) -> dict[str, DataFrame]:
        kwargs = {}
        if raw_trades is not None:
            kwargs["raw_trades"] = raw_trades
        if aggregated_trades is not None:
            kwargs["aggregated_trades"] = aggregated_trades
        if filtered_trades is not None:
            kwargs["filtered_trades"] = filtered_trades
        return kwargs

    def get_oldest_timestamp(self, trades: list[dict]) -> datetime | None:
        if trades:
            return trades[-1]["timestamp"]
        return None

    def split_trades(
        self, timestamp_from: datetime, timestamp_to: datetime, trades: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Split fetched trades into the current partition and buffered remainder."""
        unique = set()
        valid_trades = []
        buffered_trades = []
        for trade in trades:
            uid = trade["uid"]
            if uid in unique:
                continue
            unique.add(uid)
            timestamp = trade["timestamp"]
            if timestamp_from <= timestamp < timestamp_to:
                valid_trades.append(trade)
            elif timestamp < timestamp_from:
                buffered_trades.append(trade)
        return valid_trades, buffered_trades

    def parse_data(self, data: list) -> list:
        raise NotImplementedError

    def get_data_frame(self, trades: list) -> DataFrame:
        """Build a DataFrame from normalized trades in ascending order."""
        data_frame = pd.DataFrame(trades, columns=self.columns)
        # REST API, data is reverse order
        return data_frame.iloc[::-1]

    def assert_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        trades: list | None = None,
    ) -> None:
        """Assert basic integrity of a normalized trade frame."""
        # Are trades unique?
        assert len(data_frame["uid"].unique()) == len(trades)
        assert_type_decimal(data_frame, ("price", "volume", "notional"))
        # Are trades within partition?
        df = data_frame.sort_values(["timestamp", "nanoseconds", "index"])
        if len(df) >= 2:
            assert (
                timestamp_from
                <= df.iloc[0].timestamp
                <= df.iloc[-1].timestamp
                < timestamp_to
            )


class IntegerPaginationMixin:
    """Binance, ByBit, and Coinbase REST API."""

    def get_pagination_id(self, timestamp_from: datetime) -> int | None:
        """Get integer pagination_id."""
        return TradeData.objects.get_last_uid(self.symbol, timestamp_from)


class SequentialIntegerMixin(IntegerPaginationMixin):
    """Binance, ByBit, and Coinbase REST API."""

    def assert_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        trades: list | None = None,
    ) -> None:
        """Assert sequential integer data_frame."""
        super().assert_data_frame(timestamp_from, timestamp_to, data_frame, trades)
        # Missing orders.
        expected = len(trades) - 1
        diff = data_frame["index"].diff().dropna()
        assert abs(diff.sum()) == expected
