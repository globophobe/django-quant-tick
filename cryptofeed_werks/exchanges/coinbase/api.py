import time
from datetime import datetime
from typing import Callable, Optional

import httpx

from cryptofeed_werks.controllers import HTTPX_ERRORS


def get_coinbase_api_response(
    get_api_url: Callable,
    base_url: str,
    timestamp_from: Optional[datetime],
    pagination_id: Optional[str] = None,
    retry=30,
):
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
