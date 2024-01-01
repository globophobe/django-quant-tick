from datetime import datetime, timezone
from decimal import Decimal
from functools import partial
from typing import Optional

from quant_tick.controllers import iter_api
from quant_tick.lib import candles_to_data_frame

from .api import get_bybit_api_response, get_bybit_api_url
from .constants import API_URL, INVERSE_CONTRACTS, MIN_ELAPSED_PER_REQUEST

MAX_RESULTS = 200


def get_bybit_candle_api_url(api_symbol: str) -> str:
    """Get Bybit candle API url."""
    return f"{API_URL}/v5/market/kline"


def format_bybit_candle_timestamp(timestamp: datetime) -> float:
    """Format Bybit candles timestamp."""
    return int(timestamp.timestamp()) * 1000  # Milliseconds


def get_bybit_candle_pagination_id(
    timestamp: datetime, last_data: list[dict] = [], data: list[dict] = []
) -> None:
    """Get Bybit candle pagination_id."""


def get_bybit_candle_timestamp(candle: list) -> datetime:
    """Get Bybit candle timestamp."""
    timestamp = int(candle[0]) / 1000  # Milliseconds
    return datetime.fromtimestamp(timestamp).replace(tzinfo=timezone.utc)


def bybit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    interval: str = "1",
    limit: int = 60,
    log_format: Optional[str] = None,
) -> list[dict]:
    """Get Bybit candles."""
    ts_from = format_bybit_candle_timestamp(timestamp_from)
    category = "inverse" if api_symbol in INVERSE_CONTRACTS else "linear"
    params = f"symbol={api_symbol}&category={category}&start={ts_from}&interval={interval}&limit={limit}"
    url = get_bybit_candle_api_url(api_symbol)
    candles, _ = iter_api(
        f"{url}?{params}",
        get_bybit_candle_pagination_id,
        get_bybit_candle_timestamp,
        partial(get_bybit_api_response, get_bybit_api_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=timestamp_to,
        log_format=log_format,
    )
    for index, candle in enumerate(candles):
        candles[index] = {
            "timestamp": get_bybit_candle_timestamp(candle),
            "open": Decimal(candle[1]),
            "high": Decimal(candle[2]),
            "low": Decimal(candle[3]),
            "close": Decimal(candle[4]),
            "volume": Decimal(candle[5])
        }
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)
