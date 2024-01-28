import time
from collections.abc import Callable
from datetime import datetime

import httpx

from quant_tick.controllers import HTTPX_ERRORS


def get_coinbase_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: datetime | None = None,
    pagination_id: str | None = None,
    retry: int = 30,
) -> list[dict]:
    """Get Coinbase API response."""
    try:
        url = get_api_url(
            base_url, timestamp_from=timestamp_from, pagination_id=pagination_id
        )
        response = httpx.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_coinbase_api_response(
                get_api_url,
                base_url,
                timestamp_from=timestamp_from,
                pagination_id=pagination_id,
                retry=retry,
            )
        raise
