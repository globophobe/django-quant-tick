from datetime import datetime
from decimal import Decimal
from functools import partial

from pandas import DataFrame

from quant_tick.controllers import iter_api, throttle_api_requests
from quant_tick.lib import (
    candles_to_data_frame,
    parse_datetime,
    timestamp_to_inclusive,
)

from .api import format_bitfinex_api_timestamp, get_bitfinex_api_response
from .constants import (
    API_URL,
    BITFINEX_MAX_REQUESTS_RESET,
    BITFINEX_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    MAX_RESULTS,
    MIN_ELAPSED_PER_REQUEST,
)


def get_bitfinex_candle_url(url: str, pagination_id: int) -> str:
    """Get Bitfinex candle URL."""
    if pagination_id:
        url += f"&end={pagination_id}"
    return url


def get_bitfinex_candle_pagination_id(
    timestamp: datetime, last_data: list | None = None, data: list | None = None
) -> int | None:
    """Get Bitfinex candle pagination ID."""
    data = data or []
    if len(data):
        return data[-1][0]


def get_bitfinex_candle_timestamp(candle: dict) -> datetime:
    """Get Bitfinex candle timestamp."""
    return parse_datetime(candle[0], unit="ms")


def bitfinex_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    time_frame: str = "1m",
    log_format: str | None = None,
) -> DataFrame:
    """Get candles."""
    throttle_api_requests(
        BITFINEX_MAX_REQUESTS_RESET,
        BITFINEX_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    ts_to = timestamp_to_inclusive(timestamp_from, timestamp_to, value="1min")
    delta = ts_to - timestamp_from
    total_minutes = delta.total_seconds() / 60
    limit = int(total_minutes) + 1
    max_results = limit if limit <= MAX_RESULTS else MAX_RESULTS
    url = f"{API_URL}/candles/trade:{time_frame}:{api_symbol}/hist?limit={max_results}"
    results, _ = iter_api(
        url,
        get_bitfinex_candle_pagination_id,
        get_bitfinex_candle_timestamp,
        partial(get_bitfinex_api_response, get_bitfinex_candle_url),
        max_results,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=format_bitfinex_api_timestamp(ts_to),
        log_format=log_format,
    )
    candles = [
        {
            "timestamp": get_bitfinex_candle_timestamp(result),
            "open": Decimal(result[1]),
            "high": Decimal(result[3]),
            "low": Decimal(result[4]),
            "close": Decimal(result[2]),
            "notional": Decimal(result[5]),
        }
        for result in results
    ]
    # Bitfinex API may return candles less than timestamp_from
    filtered_candles = [
        candle for candle in candles if candle["timestamp"] >= timestamp_from
    ]
    return candles_to_data_frame(timestamp_from, timestamp_to, filtered_candles)
