import time

import httpx

from quant_tick.controllers import HTTPX_ERRORS

from .constants import MIN_ELAPSED_PER_REQUEST


def get_coinbase_advanced_api_response(url: str, retry: int = 30) -> dict:
    start = time.time()
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            return get_coinbase_advanced_api_response(url, retry=retry - 1)
        raise
    finally:
        elapsed = time.time() - start
        if elapsed < MIN_ELAPSED_PER_REQUEST:
            time.sleep(MIN_ELAPSED_PER_REQUEST - elapsed)
