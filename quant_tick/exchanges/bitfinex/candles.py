from datetime import datetime
from decimal import Decimal
from functools import partial

from pandas import DataFrame

from quant_tick.controllers import iter_api, throttle_api_requests
from quant_tick.lib import (
    candles_to_data_frame,
    get_interval_inclusive_end,
    get_interval_limit,
    parse_datetime,
    parse_fixed_resolution_minutes,
    resample_candles,
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

BITFINEX_TIME_FRAMES_BY_MINUTES = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    180: "3h",
    360: "6h",
    720: "12h",
    1440: "1D",
    10080: "1W",
    20160: "14D",
}

BITFINEX_SUPPORTED_TIME_FRAMES = set(BITFINEX_TIME_FRAMES_BY_MINUTES.values()) | {
    "1M"
}


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


def get_bitfinex_fetch_time_frame(
    resolution: str | int | None,
) -> tuple[int | None, str]:
    """Get the largest Bitfinex time frame that can be resampled."""
    if resolution is None:
        return 1, "1m"
    if isinstance(resolution, str):
        raw = resolution.strip()
        if raw in BITFINEX_SUPPORTED_TIME_FRAMES:
            if raw == "1M":
                return None, raw
            return parse_fixed_resolution_minutes(raw), raw
    target_minutes = parse_fixed_resolution_minutes(resolution)
    if target_minutes in BITFINEX_TIME_FRAMES_BY_MINUTES:
        return target_minutes, BITFINEX_TIME_FRAMES_BY_MINUTES[target_minutes]
    for minutes in sorted(BITFINEX_TIME_FRAMES_BY_MINUTES, reverse=True):
        if target_minutes % minutes == 0:
            return target_minutes, BITFINEX_TIME_FRAMES_BY_MINUTES[minutes]
    return target_minutes, "1m"


def fetch_bitfinex_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    time_frame: str,
    log_format: str | None = None,
) -> DataFrame:
    """Fetch Bitfinex candles."""
    ts_to = get_interval_inclusive_end(timestamp_from, timestamp_to, time_frame)
    max_results = get_interval_limit(timestamp_from, ts_to, time_frame, MAX_RESULTS)
    throttle_api_requests(
        BITFINEX_MAX_REQUESTS_RESET,
        BITFINEX_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    url = f"{API_URL}/candles/trade:{time_frame}:{api_symbol}/hist?limit={max_results}"
    results, _, _ = iter_api(
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
    filtered_candles = [
        candle for candle in candles if candle["timestamp"] >= timestamp_from
    ]
    return candles_to_data_frame(timestamp_from, timestamp_to, filtered_candles)


def bitfinex_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    time_frame: str = "1m",
    resolution: str | int | None = None,
    log_format: str | None = None,
) -> DataFrame:
    """Get candles."""
    requested = resolution if resolution is not None else time_frame
    target_minutes, fetch_time_frame = get_bitfinex_fetch_time_frame(requested)
    data_frame = fetch_bitfinex_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        time_frame=fetch_time_frame,
        log_format=log_format,
    )
    if target_minutes is None:
        return data_frame
    if parse_fixed_resolution_minutes(fetch_time_frame) == target_minutes:
        return data_frame
    return resample_candles(
        data_frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target_minutes,
    )
