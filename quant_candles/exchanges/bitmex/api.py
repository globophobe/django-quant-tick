import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from functools import partial
from typing import Callable, List, Optional

import httpx

from quant_candles.controllers import HTTPX_ERRORS
from quant_candles.lib import parse_datetime

from .constants import MAX_RESULTS

logger = logging.getLogger(__name__)


def get_bitmex_api_url(
    url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
) -> str:
    """Get BitMEX API URL."""
    url += f"&count={MAX_RESULTS}&reverse=true"
    if pagination_id:
        return url + f"&endTime={pagination_id}"
    return url


def get_bitmex_api_pagination_id(
    timestamp: datetime, last_data: List[dict] = [], data: List[dict] = []
) -> str:
    """Get BitMEX API pagination_id."""
    return format_bitmex_api_timestamp(timestamp)


def get_bitmex_api_timestamp(trade: dict) -> datetime:
    """Get BitMEX API timestamp."""
    return parse_datetime(trade["timestamp"])


def format_bitmex_api_timestamp(timestamp: datetime) -> str:
    """Format BitMEX API timestamp."""
    return timestamp.replace(tzinfo=None).isoformat()


def get_bitmex_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
    retry: int = 30,
):
    """Get BitMEX API response."""
    retry_request = partial(
        get_bitmex_api_response,
        get_api_url,
        base_url,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
    )
    try:
        url = get_api_url(
            base_url, timestamp_from=timestamp_from, pagination_id=pagination_id
        )
        response = httpx.get(url)
        if response.status_code == 200:
            remaining = response.headers["x-ratelimit-remaining"]
            reset = response.headers["x-ratelimit-reset"]
            if remaining and reset:
                remaining = int(remaining)
                reset = int(reset)
                if remaining == 0:
                    timestamp = datetime.utcnow().timestamp()
                    if reset > timestamp:
                        sleep_duration = reset - timestamp
                        time.sleep(sleep_duration)
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
            return retry_request(retry=retry)
        raise
