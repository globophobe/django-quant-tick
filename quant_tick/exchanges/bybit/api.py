import json
import time
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)

from .constants import (
    BYBIT_MAX_REQUESTS_RESET,
    BYBIT_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
)


def get_bybit_api_url(url: str, timestamp_from: datetime) -> str:
    """Get Bybit API url."""
    return url


def get_bybit_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: Optional[datetime] = None,
    pagination_id: Optional[str] = None,
    retry: int = 30,
) -> list[dict]:
    """Get Bybit API response."""
    throttle_api_requests(
        BYBIT_MAX_REQUESTS_RESET,
        BYBIT_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    try:
        response = httpx.get(get_bybit_api_url(base_url, pagination_id))
        increment_api_total_requests(BYBIT_TOTAL_REQUESTS)
        if response.status_code == 200:
            result = response.read()
            data = json.loads(result, parse_float=Decimal)
            assert data["retMsg"] == "OK"
            res = data["result"]["list"]
            # Descending order, please
            res.reverse()
            return res
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bybit_api_response(
                get_api_url, base_url, timestamp_from, pagination_id, retry
            )
        raise
