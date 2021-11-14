from datetime import datetime
from functools import partial
from typing import Optional

from cryptofeed_werks.controllers import iter_api
from cryptofeed_werks.lib import parse_datetime

from .api import get_coinbase_api_response
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_coinbase_trades_url(
    url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
):
    """Get coinbase trades URL."""
    if pagination_id:
        url = f"{url}?after={pagination_id}"
    return url


def get_coinbase_trades_pagination_id(timestamp, last_data=[], data=[]):
    """
    Pagination details: https://docs.pro.coinbase.com/#pagination
    """
    if len(data):
        return data[-1]["trade_id"]


def get_coinbase_trades_timestamp(trade):
    return parse_datetime(trade["time"])


def get_trades(symbol, timestamp_from, pagination_id, log_format=None):
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
