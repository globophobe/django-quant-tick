from datetime import datetime
from functools import partial
from typing import List, Optional

from quant_candles.controllers import iter_api
from quant_candles.lib import parse_datetime

from .api import format_ftx_api_timestamp, get_ftx_api_response
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_ftx_trades_url(
    url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
):
    """Get FTX trades URL.

    Exclude start time, for is_last_iteration.
    """
    url += f"?limit={MAX_RESULTS}"
    if pagination_id:
        return url + f"&end_time={pagination_id}"
    return url


def get_ftx_trades_pagination_id(
    timestamp: datetime, last_data: List[dict] = [], data: List[dict] = []
):
    """Get FTX trades pagination_id.

    FTX API is seriously donkey balls. If more than 200 trades at same timestamp,
    the next timestamp is same as first. In such case, the pagination_id needs
    to be adjusted a small amount until unique trades are captured again.
    """
    pagination_id = format_ftx_api_timestamp(timestamp)
    ids = [trade["id"] for trade in last_data]
    unique = [trade["id"] for trade in data if trade["id"] not in ids]
    if len(data) == MAX_RESULTS and not len(unique):
        pagination_id -= 1e-6
        return round(pagination_id, 6)
    return pagination_id


def get_ftx_trades_timestamp(trade: dict) -> datetime:
    """Get FTX trades timestamp."""
    return parse_datetime(trade["time"])


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: str,
    log_format: Optional[str] = None,
):
    """Get trades."""
    url = f"{API_URL}/markets/{symbol}/trades"
    return iter_api(
        url,
        get_ftx_trades_pagination_id,
        get_ftx_trades_timestamp,
        partial(get_ftx_api_response, get_ftx_trades_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
