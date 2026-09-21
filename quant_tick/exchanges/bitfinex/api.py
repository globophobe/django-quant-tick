import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx2

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)

from .constants import (
    BITFINEX_MAX_REQUESTS_RESET,
    BITFINEX_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    TRADE_MAX_RESULTS,
)

logger = logging.getLogger(__name__)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def format_bitfinex_api_timestamp(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * 1000)  # Millisecond


def get_bitfinex_api_url(url: str, pagination_id: int) -> str:
    if pagination_id:
        return url + f"&end={pagination_id}"
    return url


def get_bitfinex_api_pagination_id(
    timestamp: datetime, last_data: list | None = None, data: list | None = None
) -> int | None:
    data = data or []
    if len(data):
        last_trade = data[-1]
        last_id = last_trade[1]
        # Is data fetched same as previous?
        if len(data) == TRADE_MAX_RESULTS and last_data and last_id == last_data[-1][1]:
            return None
        if len(data):
            return last_id


def get_bitfinex_api_timestamp(trade: dict) -> datetime:
    return EPOCH + timedelta(milliseconds=int(trade[1]))


def get_bitfinex_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: datetime | None = None,
    pagination_id: str | None = None,
    retry: int = 30,
) -> list[dict]:
    """Count every attempt; HTTP 429 and transport errors share the retry budget."""
    url = get_api_url(base_url, pagination_id=pagination_id)
    for attempt in range(retry + 1):
        throttle_api_requests(
            BITFINEX_MAX_REQUESTS_RESET,
            BITFINEX_TOTAL_REQUESTS,
            MAX_REQUESTS_RESET,
            MAX_REQUESTS,
        )
        try:
            increment_api_total_requests(BITFINEX_TOTAL_REQUESTS)
            response = httpx2.get(url)
            if response.status_code == 200:
                result = response.read()
                return json.loads(result, parse_float=Decimal)
            if response.status_code == 429 and attempt < retry:
                sleep_duration = response.headers.get("Retry-After", 1)
                logger.info(f"HTTP 429, sleeping {sleep_duration} seconds")
                time.sleep(int(sleep_duration))
                continue
            response.raise_for_status()
            break
        except HTTPX_ERRORS:
            if attempt == retry:
                raise
            time.sleep(1)
