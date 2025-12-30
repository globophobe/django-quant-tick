from datetime import datetime
from decimal import Decimal
from functools import partial

from pandas import DataFrame

from quant_tick.controllers import iter_api
from quant_tick.lib import candles_to_data_frame, parse_datetime, timestamp_to_inclusive

from .api import format_binance_api_timestamp, get_binance_api_response
from .constants import MAX_RESULTS, MIN_ELAPSED_PER_REQUEST, SPOT_API_URL


def get_binance_candle_url(
    url: str, timestamp_from: datetime, pagination_id: int
) -> DataFrame:
    """Get Binance candle URL."""
    if pagination_id:
        return url + f"&endTime={pagination_id}"
    return url


def get_binance_candle_pagination_id(
    timestamp: datetime,
    last_data: list | None = None,
    data: list | None = None,
) -> str | None:
    """Get Binance candle pagination_id."""
    data = data or []
    if len(data):
        return data[-1][0]


def get_binance_candle_timestamp(candle: list) -> datetime:
    """Get Binance candle timestamp."""
    return parse_datetime(candle[0], unit="ms")


def binance_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    interval: str = "1m",
    limit: int | None = None,
    log_format: str | None = None,
) -> DataFrame:
    """Get coinbase candles."""
    ts_from = format_binance_api_timestamp(timestamp_from)
    ts_to = format_binance_api_timestamp(
        timestamp_to_inclusive(timestamp_from, timestamp_to, value="1min")
    )
    url = (
        f"{SPOT_API_URL}/klines?symbol={api_symbol}&interval={interval}&startTime={ts_from}"
    )
    candles, _ = iter_api(
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
