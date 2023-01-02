from datetime import datetime
from functools import partial
from typing import Optional

from quant_candles.controllers import iter_api

from .api import (
    get_bitfinex_api_pagination_id,
    get_bitfinex_api_response,
    get_bitfinex_api_timestamp,
    get_bitfinex_api_url,
)
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: int,
    log_format: Optional[str] = None,
):
    """Get trades."""
    # No start query param
    # Specifying start, end returns MAX_RESULTS
    url = f"{API_URL}/trades/{symbol}/hist?limit={MAX_RESULTS}"
    return iter_api(
        url,
        get_bitfinex_api_pagination_id,
        get_bitfinex_api_timestamp,
        partial(get_bitfinex_api_response, get_bitfinex_api_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
