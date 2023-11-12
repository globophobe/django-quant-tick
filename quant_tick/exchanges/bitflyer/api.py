import json
import time
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

import httpx

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    iter_api,
    throttle_api_requests,
)
from quant_tick.lib import parse_datetime

from .constants import (
    BITFLYER_MAX_REQUESTS_RESET,
    BITFLYER_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    MAX_RESULTS,
    MIN_ELAPSED_PER_REQUEST,
    URL,
)


def get_bitflyer_api_url(url: str, pagination_id: int) -> str:
    """Get Bitflyer API URL."""
    if pagination_id:
        url += f"{url}&before={pagination_id}"
    return url


def get_bitflyer_api_pagination_id(
    timestamp: datetime, last_data: list = [], data: list = []
) -> Optional[int]:
    """Get Bitflyer API pagination_id."""
    if len(data):
        return data[-1]["id"]


def get_bitflyer_api_timestamp(trade):
    """Get Bitflyer API timestamp."""
    return parse_datetime(trade["exec_date"])


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: int,
    log_format: Optional[str] = None,
) -> List[dict]:
    """Get trades."""
    url = f"{URL}/executions?product_code={symbol}&count={MAX_RESULTS}"
    return iter_api(
        url,
        get_bitflyer_api_pagination_id,
        get_bitflyer_api_timestamp,
        get_bitflyer_api_response,
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )


def get_bitflyer_api_response(
    url: str, pagination_id: Optional[int] = None, retry: int = 30
) -> List[dict]:
    """Get Bitflyer API response."""
    throttle_api_requests(
        BITFLYER_MAX_REQUESTS_RESET,
        BITFLYER_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    try:
        response = httpx.get(get_bitflyer_api_url(url, pagination_id))
        increment_api_total_requests(BITFLYER_TOTAL_REQUESTS)
        if response.status_code == 200:
            result = response.read()
            data = json.loads(result, parse_float=Decimal)
            return data
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bitflyer_api_response(url, pagination_id, retry)
        raise
