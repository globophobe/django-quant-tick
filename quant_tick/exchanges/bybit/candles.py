from datetime import datetime, timezone
from decimal import Decimal
from functools import partial
from typing import List, Optional

from quant_tick.controllers import iter_api
from quant_tick.lib import candles_to_data_frame

from .api import get_bybit_api_response, get_bybit_api_url
from .constants import API_URL, INVERSE_CONTRACTS, MIN_ELAPSED_PER_REQUEST

MAX_RESULTS = 200


def get_bybit_candle_api_url(api_symbol: str) -> str:
    """Get Bybit candle API url."""
    if api_symbol in INVERSE_CONTRACTS:
        return f"{API_URL}/v2/public/kline/list"
    else:
        return f"{API_URL}/public/linear/kline"


def format_bybit_candle_timestamp(timestamp: datetime) -> float:
    """Format Bybit candles timestamp."""
    return int(timestamp.timestamp())


def get_bybit_candle_pagination_id(
    timestamp: datetime, last_data: List[dict] = [], data: List[dict] = []
) -> None:
    """Get Bybit candle pagination_id."""


def get_bybit_candle_timestamp(candle: dict) -> datetime:
    """Get Bybit candle timestamp."""
    return datetime.fromtimestamp(candle["open_time"]).replace(tzinfo=timezone.utc)


def bybit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    interval: str = "1",
    limit: int = 60,
    log_format: Optional[str] = None,
) -> List[dict]:
    """Get Bybit candles."""
    ts_from = format_bybit_candle_timestamp(timestamp_from)
    params = f"symbol={api_symbol}&from={ts_from}&interval={interval}&limit={limit}"
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
    for candle in candles:
        candle["timestamp"] = get_bybit_candle_timestamp(candle)
        candle["volume"] = Decimal(candle["volume"])
        keys = ("timestamp", "open", "high", "low", "close", "volume", "notional")
        for key in [key for key in candle if key not in keys]:
            del candle[key]
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)
