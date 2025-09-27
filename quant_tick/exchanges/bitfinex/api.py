import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from functools import partial

import httpx

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)
from quant_tick.lib import parse_datetime

from .constants import (
    BITFINEX_MAX_REQUESTS_RESET,
    BITFINEX_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    MAX_RESULTS,
)

logger = logging.getLogger(__name__)


def format_bitfinex_api_timestamp(timestamp: datetime) -> int:
    """Format Bitfinex API timestmap."""
    return int(timestamp.timestamp() * 1000)  # Millisecond


def get_bitfinex_api_url(url: str, pagination_id: int) -> str:
    """Get Bitfinex API URL."""
    if pagination_id:
        return url + f"&end={pagination_id}"
    return url


def get_bitfinex_api_pagination_id(
    timestamp: datetime, last_data: list | None = None, data: list | None = None
) -> int | None:
    """Get Bitfinex API pagination ID."""
    data = data or []
    if len(data):
        last_trade = data[-1]
        last_id = last_trade[1]
        # Is data fetched same as previous?
        if len(data) == MAX_RESULTS and last_data and last_id == last_data[-1][1]:
            return None
        if len(data):
            return last_id


def get_bitfinex_api_timestamp(trade: dict) -> datetime:
    """Get Bitfinex API timestamp."""
    return parse_datetime(trade[1], unit="ms")


def get_bitfinex_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: datetime | None = None,
    pagination_id: str | None = None,
    retry: int = 30,
) -> list[dict]:
    """Get Bitfinex API response."""
    throttle_api_requests(
        BITFINEX_MAX_REQUESTS_RESET,
        BITFINEX_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    retry_request = partial(
        get_bitfinex_api_response,
        get_api_url,
        base_url,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
    )
    try:
        url = get_api_url(base_url, pagination_id=pagination_id)
        response = httpx.get(url)
        increment_api_total_requests(BITFINEX_TOTAL_REQUESTS)
        if response.status_code == 200:
            result = response.read()
            return json.loads(result, parse_float=Decimal)
        elif response.status_code == 429:
            sleep_duration = response.headers.get("Retry-After", 1)
            logger.info(f"HTTP 429, sleeping {sleep_duration} seconds")
            time.sleep(int(sleep_duration))
            return retry_request(retry=retry)
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bitfinex_api_response(
                get_api_url,
                base_url,
                timestamp_from=timestamp_from,
                pagination_id=pagination_id,
                retry=retry,
            )
        raise
