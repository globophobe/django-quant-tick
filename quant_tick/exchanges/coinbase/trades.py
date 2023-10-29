from datetime import datetime
from functools import partial
from typing import List, Optional

from quant_tick.controllers import iter_api
from quant_tick.lib import parse_datetime

from .api import get_coinbase_api_response
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_coinbase_trades_url(
    url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[int] = None,
) -> str:
    """Get coinbase trades URL."""
    if pagination_id:
        url = f"{url}?after={pagination_id}"
    return url


def get_coinbase_trades_pagination_id(
    timestamp: datetime, last_data: list = [], data: list = []
) -> Optional[int]:
    """Get Coinbase trades pagination_id.

    Pagination details: https://docs.pro.coinbase.com/#pagination
    """
    if len(data):
        return data[-1]["trade_id"]


def get_coinbase_trades_timestamp(trade: dict) -> datetime:
    """Get Coinbase trades timestamp."""
    return parse_datetime(trade["time"])


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: int,
    log_format: Optional[str] = None,
) -> List[dict]:
    """Get trades."""
    url = f"{API_URL}/products/{symbol}/trades"
    return iter_api(
        url,
        get_coinbase_trades_pagination_id,
        get_coinbase_trades_timestamp,
        partial(get_coinbase_api_response, get_coinbase_trades_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
