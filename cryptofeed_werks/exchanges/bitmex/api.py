import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Callable, List, Optional

import httpx

from cryptofeed_werks.controllers import HTTPX_ERRORS
from cryptofeed_werks.lib import parse_datetime

from .constants import MAX_RESULTS


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
):
    """Get BitMEX API pagination_id."""
    return format_bitmex_api_timestamp(timestamp)


def get_bitmex_api_timestamp(trade: dict):
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
    try:
        url = get_api_url(
            base_url, timestamp_from=timestamp_from, pagination_id=pagination_id
        )
        response = httpx.get(url)
        if response.status_code == 200:
            remaining = response.headers["x-ratelimit-remaining"]
            if remaining == 0:
                timestamp = datetime.utcnow().timestamp()
                reset = response.headers["x-ratelimit-reset"]
                if reset > timestamp:
                    sleep_duration = reset - timestamp
                    print(f"Max requests, sleeping {sleep_duration} seconds")
                    time.sleep(sleep_duration)
            result = response.read()
            return json.loads(result, parse_float=Decimal)
        elif response.status_code == 429:
            retry = response.headers.get("Retry-After", 1)
            time.sleep(int(retry))
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bitmex_api_response(
                get_api_url,
                base_url,
                timestamp_from=timestamp_from,
                pagination_id=pagination_id,
                retry=retry,
            )
        raise
