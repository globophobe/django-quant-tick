from datetime import datetime
from functools import partial
from typing import Optional

from quant_werks.controllers import iter_api

from .api import (
    get_bitmex_api_pagination_id,
    get_bitmex_api_response,
    get_bitmex_api_timestamp,
    get_bitmex_api_url,
)
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: str,
    log_format: Optional[str] = None,
):
    """Get trades."""
    url = f"{API_URL}/trade?symbol={symbol}"
    return iter_api(
        url,
        get_bitmex_api_pagination_id,
        get_bitmex_api_timestamp,
        partial(get_bitmex_api_response, get_bitmex_api_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
