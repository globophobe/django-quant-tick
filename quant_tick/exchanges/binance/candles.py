from datetime import datetime
from decimal import Decimal
from functools import partial

from pandas import DataFrame

from quant_tick.controllers import iter_api
from quant_tick.lib import (
    candles_to_data_frame,
    get_interval_inclusive_end,
    parse_datetime,
    parse_fixed_resolution_minutes,
)

from .api import format_binance_api_timestamp, get_binance_api_response
from .constants import CANDLE_MAX_RESULTS, MIN_ELAPSED_PER_REQUEST, SPOT_API_URL

BINANCE_INTERVALS_BY_MINUTES = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    360: "6h",
    480: "8h",
    720: "12h",
    1440: "1d",
    4320: "3d",
    10080: "1w",
}

BINANCE_SUPPORTED_INTERVALS = set(BINANCE_INTERVALS_BY_MINUTES.values()) | {
    "1s",
    "1M",
}


def parse_binance_candle_resolution(value: str | int | None) -> int:
    """Parse a Binance candle resolution into whole minutes."""
    return parse_fixed_resolution_minutes(value)


def get_binance_interval(resolution: str | int | None) -> str:
    """Map requested resolution to a supported Binance interval."""
    if isinstance(resolution, str):
        raw = resolution.strip()
        if raw in BINANCE_SUPPORTED_INTERVALS:
            return raw
    minutes = parse_binance_candle_resolution(resolution)
    try:
        return BINANCE_INTERVALS_BY_MINUTES[minutes]
    except KeyError as exc:
        raise ValueError(f"unsupported Binance candle resolution: {resolution}") from exc


def get_binance_candle_url(
    url: str, timestamp_from: datetime, pagination_id: int
) -> str:
    if pagination_id:
        return url + f"&endTime={pagination_id}"
    return url


def get_binance_candle_pagination_id(
    timestamp: datetime,
    last_data: list | None = None,
    data: list | None = None,
) -> int:
    return format_binance_api_timestamp(timestamp) - 1


def get_binance_candle_timestamp(candle: list) -> datetime:
    return parse_datetime(candle[0], unit="ms")


def fetch_binance_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    interval: str,
    limit: int = CANDLE_MAX_RESULTS,
    log_format: str | None = None,
) -> DataFrame:
    """Fetch Binance candles."""
    url = f"{SPOT_API_URL}/klines?symbol={api_symbol}&interval={interval}&limit={limit}"
    ts_to = get_interval_inclusive_end(timestamp_from, timestamp_to, interval)
    results, _, _ = iter_api(
        url,
        get_binance_candle_pagination_id,
        get_binance_candle_timestamp,
        partial(get_binance_api_response, get_binance_candle_url),
        limit,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=format_binance_api_timestamp(ts_to),
        log_format=log_format,
    )
    candles = [
        {
            "timestamp": get_binance_candle_timestamp(result),
            "open": Decimal(str(result[1])),
            "high": Decimal(str(result[2])),
            "low": Decimal(str(result[3])),
            "close": Decimal(str(result[4])),
            "notional": Decimal(str(result[5])),
        }
        for result in results
    ]
    filtered_candles = [
        candle for candle in candles if candle["timestamp"] >= timestamp_from
    ]
    return candles_to_data_frame(timestamp_from, timestamp_to, filtered_candles)


def binance_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    interval: str = "1m",
    resolution: str | int | None = None,
    limit: int | None = None,
    log_format: str | None = None,
) -> DataFrame:
    interval = get_binance_interval(resolution if resolution is not None else interval)
    return fetch_binance_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        interval=interval,
        limit=int(limit or CANDLE_MAX_RESULTS),
        log_format=log_format,
    )
