import os
import time
from datetime import datetime
from decimal import Decimal
from operator import eq
from typing import Callable, Optional

import pandas as pd
from pandas import DataFrame

from cryptofeed_werks.lib import (
    aggregate_trades,
    assert_type_decimal,
    validate_data_frame,
    volume_filter_with_time_window,
)
from cryptofeed_werks.models import Candle, Symbol

from .base import BaseController


def iter_api(
    url: str,
    get_api_pagination_id: Callable,
    get_api_timestamp: Callable,
    get_api_response: Callable,
    max_results: int,
    min_elapsed_per_request,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
    log_format: Optional[str] = None,
) -> list:
    """Iterate exchange API."""
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
                print(log_format.format(**{"timestamp": t}))
        # Throttle requests
        elapsed = time.time() - start
        if elapsed < min_elapsed_per_request:
            time.sleep(min_elapsed_per_request - elapsed)
    return results, is_last_iteration


def get_api_max_requests_reset(seconds: int) -> float:
    return time.time() + seconds


def set_api_environ_vars(
    max_requests_reset_key: str,
    total_requests_key: str,
    max_requests_reset: float,
    reset: bool = False,
) -> None:
    """Set API environment variables."""
    if reset or (max_requests_reset_key not in os.environ):
        max_requests_reset = get_api_max_requests_reset(max_requests_reset)
        os.environ[max_requests_reset_key] = str(max_requests_reset)
    if reset or (total_requests_key not in os.environ):
        os.environ[total_requests_key] = str(0)


def increment_api_total_requests(total_requests_key: str) -> None:
    """Increment API total requests."""
    total_requests = int(os.environ[total_requests_key])
    os.environ[total_requests_key] = str(total_requests + 1)


def throttle_api_requests(
    max_requests_reset_key: str,
    total_requests_key: str,
    max_requests_reset: float,
    max_requests: int,
) -> None:
    """Throttle API requests."""
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
                print(f"Max requests, sleeping {sleep_time} seconds")
                time.sleep(sleep_time)


class ExchangeREST(BaseController):
    def get_pagination_id(self, timestamp_to: datetime):
        """Get pagination_id for symbol."""
        raise NotImplementedError

    def iter_api(self, symbol: Symbol, pagination_id: str, log_format: str):
        """Iter exchange API."""
        raise NotImplementedError

    def get_max_timestamp_to(self) -> datetime:
        """Get max timestamp_to, to omit incomplete candles."""

    def get_timestamp_to(
        self, timestamp_to: datetime, max_timestamp_to: datetime
    ) -> datetime:
        """Get timestamp_to, omitting first incomplete candle."""

    def main(self) -> DataFrame:
        """Main loop, with Candle.iter_all"""
        for timestamp_from, timestamp_to in Candle.iter_all(
            self.symbol,
            self.timestamp_from,
            self.timestamp_to,
            reverse=True,
            retry=self.retry,
        ):
            pagination_id = self.get_pagination_id(timestamp_to)
            trade_data, is_last_iteration = self.iter_api(timestamp_from, pagination_id)
            trades = self.parse_data(trade_data)
            valid_trades = self.get_valid_trades(timestamp_from, timestamp_to, trades)
            data_frame = self.get_data_frame(valid_trades)
            candles = self.get_candles(timestamp_from, timestamp_to)
            # Are there any trades?
            if len(data_frame):
                self.assert_data_frame(
                    timestamp_from, timestamp_to, data_frame, valid_trades
                )
                aggregated = aggregate_trades(data_frame)
                filtered = volume_filter_with_time_window(
                    aggregated, min_volume=self.symbol.min_volume
                )
            else:
                filtered = pd.DataFrame([])
            validated = validate_data_frame(
                timestamp_from, timestamp_to, filtered, candles
            )
            self.on_data_frame(
                self.symbol, timestamp_from, timestamp_to, filtered, validated=validated
            )
            # Complete
            if is_last_iteration:
                break

    def parse_data(self, data: list) -> list:
        """Parse trade data."""
        return [
            {
                "uid": self.get_uid(trade),
                "timestamp": self.get_timestamp(trade),
                "nanoseconds": self.get_nanoseconds(trade),
                "price": self.get_price(trade),
                "volume": self.get_volume(trade),
                "notional": self.get_notional(trade),
                "tickRule": self.get_tick_rule(trade),
                "index": self.get_index(trade),
            }
            for trade in data
        ]

    def get_valid_trades(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        trades: list,
        operator: Callable = eq,
    ):
        """Get valid trades."""
        unique = set()
        valid_trades = []
        for trade in trades:
            is_unique = trade["uid"] not in unique
            is_within_partition = timestamp_from <= trade["timestamp"] <= timestamp_to
            if is_unique and is_within_partition:
                valid_trades.append(trade)
                unique.add(trade["uid"])
        return valid_trades

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        raise NotImplementedError

    def get_timestamp(self, trade: dict) -> datetime:
        """Get timestamp."""
        raise NotImplementedError

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        raise NotImplementedError

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        raise NotImplementedError

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        raise NotImplementedError

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule."""
        raise NotImplementedError

    def get_index(self, trade: dict) -> int:
        """Get index."""
        raise NotImplementedError

    def get_data_frame(self, trades: list) -> DataFrame:
        """Get data_frame."""
        data_frame = pd.DataFrame(trades, columns=self.columns)
        # REST API, data is reverse order
        return data_frame.iloc[::-1]

    def assert_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        trades: Optional[list] = None,
    ) -> None:
        """Assertions for data_frame."""
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
                <= timestamp_to
            )


class IntegerPaginationMixin:
    """Binance, ByBit, and Coinbase REST API."""

    def get_pagination_id(self, timestamp_from: datetime) -> Optional[int]:
        """Get integer pagination_id."""
        return Candle.get_last_uid(self.symbol, timestamp_from)


class SequentialIntegerMixin(IntegerPaginationMixin):
    """Binance, ByBit, and Coinbase REST API."""

    def assert_data_frame(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        trades: Optional[list] = None,
    ) -> None:
        """Assert sequential integer data_frame."""
        super().assert_data_frame(timestamp_from, timestamp_to, data_frame, trades)
        # Missing orders.
        expected = len(trades) - 1
        diff = data_frame["index"].diff().dropna()
        assert abs(diff.sum()) == expected


class ExchangeMultiSymbolREST(ExchangeREST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbols = self.get_symbols()

    @property
    def active_symbols(self) -> list:
        """Active symbols for root symbol."""
        return [s for s in self.symbols if (s["expiry"] >= self.timestamp_from)]

    def get_symbols(self):
        """Get symbols for root symbol."""
        raise NotImplementedError

    def main(self, timestamp_from: datetime, timestamp_to: datetime) -> DataFrame:
        for timestamp_from, timestamp_to in self.iter_timeframe:
            trades = []
            is_last_iteration = []
            # Iterate symbols
            for active_symbol in self.active_symbols:
                pagination_id = self.get_pagination_id()
                symbol = active_symbol["symbol"]
                log_format = f"{self.exchange_display} {symbol}"
                t, is_last = self.iter_api(
                    active_symbol["api_symbol"], pagination_id, log_format
                )
                trades += self.parse_data(
                    t, active_symbol["symbol"], active_symbol["expiry"]
                )
                is_last_iteration.append(is_last)
            valid_trades = self.get_valid_trades(trades)

            data_frame = self.get_data_frame(valid_trades)
            self.on_data_frame(data_frame)

            # Complete
            if all(is_last_iteration):
                break

    def parse_data(self, data: list, symbol: str, expiry: datetime) -> list:
        """Parse trade data."""
        return [
            {
                "uid": self.get_uid(trade),
                "symbol": symbol,
                "expiry": expiry,
                "timestamp": self.get_timestamp(trade),
                "nanoseconds": self.get_nanoseconds(trade),
                "price": self.get_price(trade),
                "volume": self.get_volume(trade),
                "notional": self.get_notional(trade),
                "tickRule": self.get_tick_rule(trade),
                "index": self.get_index(trade),
            }
            for trade in data
        ]
