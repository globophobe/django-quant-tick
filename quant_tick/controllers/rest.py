import logging
import os
import time
from collections.abc import Callable
from datetime import datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import assert_type_decimal
from quant_tick.models import Symbol, TradeData

from .base import BaseController
from .iterators import TradeDataIterator

logger = logging.getLogger(__name__)


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
        for timestamp_from, timestamp_to in TradeDataIterator(self.symbol).iter_all(
            self.timestamp_from,
            self.timestamp_to,
            retry=self.retry,
        ):
            buffered_trades = [
                trade
                for trade in buffered_trades
                if trade["timestamp"] < timestamp_to
            ]
            if pagination_id is None:
                pagination_id = self.get_pagination_id(timestamp_to)
            oldest_timestamp = self.get_oldest_timestamp(buffered_trades)
            while (
                not is_last_iteration
                and (oldest_timestamp is None or oldest_timestamp > timestamp_from)
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
            candles = self.get_candles(timestamp_from, timestamp_to)
            if len(data_frame):
                self.assert_data_frame(
                    timestamp_from, timestamp_to, data_frame, valid_trades
                )
            self.on_data_frame(
                self.symbol, timestamp_from, timestamp_to, data_frame, candles
            )
            # Complete
            if is_last_iteration and not buffered_trades:
                break

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
