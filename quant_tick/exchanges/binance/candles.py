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
from .constants import MAX_RESULTS, MIN_ELAPSED_PER_REQUEST, SPOT_API_URL

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
) -> DataFrame:
    if pagination_id:
        return url + f"&endTime={pagination_id}"
    return url


def get_binance_candle_pagination_id(
    timestamp: datetime,
    last_data: list | None = None,
    data: list | None = None,
) -> str | None:
    data = data or []
    if len(data):
        return data[-1][0]


def get_binance_candle_timestamp(candle: list) -> datetime:
    return parse_datetime(candle[0], unit="ms")


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
    ts_from = format_binance_api_timestamp(timestamp_from)
    ts_to = format_binance_api_timestamp(
        get_interval_inclusive_end(timestamp_from, timestamp_to, interval)
    )
    url = (
        f"{SPOT_API_URL}/klines?symbol={api_symbol}&interval={interval}&startTime={ts_from}"
    )
    candles, _, _ = iter_api(
        url,
        get_binance_candle_pagination_id,
        get_binance_candle_timestamp,
        partial(get_binance_api_response, get_binance_candle_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=ts_to,
        log_format=log_format,
    )
    c = [
        {
            "timestamp": get_binance_candle_timestamp(candle),
            "open": Decimal(str(candle[1])),
            "high": Decimal(str(candle[2])),
            "low": Decimal(str(candle[3])),
            "close": Decimal(str(candle[4])),
            "notional": Decimal(str(candle[5])),
        }
        for candle in candles
    ]
    return candles_to_data_frame(timestamp_from, timestamp_to, c)
