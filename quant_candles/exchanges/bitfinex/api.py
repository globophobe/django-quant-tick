import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Callable, List, Optional

import httpx

from quant_candles.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)
from quant_candles.lib import parse_datetime

from .constants import (
    BITFINEX_MAX_REQUESTS_RESET,
    BITFINEX_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
)


def format_bitfinex_api_timestamp(timestamp: datetime) -> int:
    """Format Bitfinex API timestmap."""
    return int(timestamp.timestamp() * 1000)  # Millisecond


def get_bitfinex_api_url(url: str, pagination_id: int) -> str:
    """Get Bitfinex API URL."""
    if pagination_id:
        return url + f"&end={pagination_id}"
    return url


def get_bitfinex_api_pagination_id(
    timestamp: datetime, last_data: list = [], data: list = []
):
    """Get Bitfinex API pagination ID."""
    if len(data):
        return data[-1][1]


def get_bitfinex_api_timestamp(trade: dict):
    """Get Bitfinex API timestamp."""
    return parse_datetime(trade[1], unit="ms")


def get_bitfinex_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
    retry: int = 30,
) -> List[dict]:
    """Get Bitfinex API response."""
    throttle_api_requests(
        BITFINEX_MAX_REQUESTS_RESET,
        BITFINEX_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    try:
        url = get_api_url(base_url, pagination_id=pagination_id)
        response = httpx.get(url)
        increment_api_total_requests(BITFINEX_TOTAL_REQUESTS)
        if response.status_code == 200:
            result = response.read()
            return json.loads(result, parse_float=Decimal)
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
