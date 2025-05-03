import logging
import os
import time
from datetime import datetime, timedelta

import httpx
from decouple import config

from quant_tick.controllers import HTTPX_ERRORS
from quant_tick.lib import get_current_time

from .constants import BINANCE_API_KEY, BINANCE_MAX_WEIGHT, MAX_WEIGHT

logger = logging.getLogger(__name__)


def format_binance_api_timestamp(timestamp: datetime) -> int:
    """Format Binance API timestmap."""
    return int(timestamp.timestamp() * 1000)  # Millisecond


def get_binance_api_sleep_duration() -> float:
    """Get Binance API sleep duration."""
    now = get_current_time()
    current_minute = now.replace(second=0, microsecond=0)
    next_minute = current_minute + timedelta(minutes=1)
    delta = next_minute - now
    return delta.total_seconds()


def get_binance_api_response(
    get_api_url: str,
    base_url: str,
    timestamp_from: datetime | None = None,
    pagination_id: int | None = None,
    retry: int = 30,
) -> list[dict]:
    """Get Binance API response."""
    try:
        headers = {"X-MBX-APIKEY": config(BINANCE_API_KEY)}
        url = get_api_url(
            base_url, timestamp_from=timestamp_from, pagination_id=pagination_id
        )
        response = httpx.get(url, headers=headers)
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
            return get_binance_api_response(
                get_api_url,
                base_url,
                timestamp_from=timestamp_from,
                pagination_id=pagination_id,
                retry=retry,
            )
        raise
