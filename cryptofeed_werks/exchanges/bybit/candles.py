from datetime import datetime, timezone
from functools import partial
from typing import List, Optional

from cryptofeed_werks.controllers import iter_api
from cryptofeed_werks.lib import candles_to_data_frame

from .api import get_bybit_api_response, get_bybit_api_url
from .constants import API_URL, MIN_ELAPSED_PER_REQUEST

MAX_RESULTS = 200


def format_bybit_candle_timestamp(timestamp: datetime) -> float:
    """Format Bybit candles timestamp."""
    return int(timestamp.timestamp())


def get_bybit_candle_pagination_id(
    timestamp: datetime, last_data: List[dict] = [], data: List[dict] = []
) -> None:
    """Get Bybit candle pagination_id."""


def get_bybit_candle_timestamp(candle: dict) -> None:
    """Get Bybit candle timestamp."""
    return datetime.fromtimestamp(candle["open_time"]).replace(tzinfo=timezone.utc)


def bybit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    interval: str = "1",
    limit: int = 60,
    log_format: Optional[str] = None,
):
    ts_from = format_bybit_candle_timestamp(timestamp_from)
    params = f"symbol={api_symbol}&from={ts_from}&interval={interval}&limit={limit}"
    url = f"{API_URL}/kline/list?{params}"
    candles, _ = iter_api(
        url,
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
        for key in ("symbol", "interval"):
            del candle[key]
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)
