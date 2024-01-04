import logging
import os
import time
from datetime import datetime, timedelta

import httpx

from quant_tick.controllers import HTTPX_ERRORS, iter_api
from quant_tick.lib import get_current_time, parse_datetime

from .constants import (
    API_URL,
    BINANCE_API_KEY,
    BINANCE_MAX_WEIGHT,
    MAX_RESULTS,
    MAX_WEIGHT,
    MIN_ELAPSED_PER_REQUEST,
)

logger = logging.getLogger(__name__)


def get_binance_api_sleep_duration() -> float:
    """Get Binance API sleep duration."""
    now = get_current_time()
    current_minute = now.replace(second=0, microsecond=0)
    next_minute = current_minute + timedelta(minutes=1)
    delta = next_minute - now
    return delta.total_seconds()


def get_binance_api_url(url: str, pagination_id: int) -> str:
    """Get Binance API url."""
    if pagination_id:
        return url + f"&fromId={pagination_id}"
    return url


def get_binance_api_pagination_id(
    timestamp: datetime, last_data: list | None = None, data: list | None = None
) -> int:
    """Get Binance API pagination_id."""
    data = data or []
    # Like bybit, binance pagination feels like an IQ test.
    if len(data):
        last_trade = data[-1]
        last_id = last_trade["id"]
        pagination_id = last_id - len(data)
        # Is it the last_id? If so, stop_iteration
        if last_id == 1:
            return None
        # Calculated pagination_id will be negative if remaining trades is
        # less than MAX_RESULTS.
        elif pagination_id <= 0:
            return 1
        else:
            return pagination_id


def get_binance_api_timestamp(trade: dict) -> datetime:
    """Get Binance API timestamp."""
    return parse_datetime(trade["time"], unit="ms")


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: int,
    log_format: str | None = None,
) -> list[dict]:
    """Get trades."""
    url = f"{API_URL}/historicalTrades?symbol={symbol}&limit={MAX_RESULTS}"
    os.environ[BINANCE_MAX_WEIGHT] = str(1195)
    result = iter_api(
        url,
        get_binance_api_pagination_id,
        get_binance_api_timestamp,
        get_binance_api_response,
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
    del os.environ[BINANCE_MAX_WEIGHT]
    return result


def get_binance_api_response(
    url: str, pagination_id: int | None = None, retry: int = 30
) -> list[dict]:
    """Get Binance API response."""
    try:
        headers = {"X-MBX-APIKEY": os.environ.get(BINANCE_API_KEY, None)}
        response = httpx.get(get_binance_api_url(url, pagination_id), headers=headers)
        if response.status_code == 200:
            # Response 429, when x-mbx-used-weight-1m is 1200
            weight = response.headers.get("x-mbx-used-weight-1m", 0)
            max_weight = os.environ.get(BINANCE_MAX_WEIGHT, MAX_WEIGHT)
            if int(weight) >= int(max_weight):
                sleep_duration = get_binance_api_sleep_duration()
                logger.info(f"Max requests, sleeping {sleep_duration} seconds")
                time.sleep(sleep_duration)
            data = response.json()
            data.reverse()  # Descending order, please
            return data
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_binance_api_response(url, pagination_id, retry)
        raise
