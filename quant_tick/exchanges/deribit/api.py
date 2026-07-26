import time
from datetime import UTC, datetime

import httpx

from quant_tick.controllers import HTTPX_ERRORS

from .constants import API_URL, MIN_ELAPSED_PER_REQUEST


def to_millis(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return int(timestamp.timestamp() * 1000)


def get_deribit_result(method: str, params: dict, retry: int = 30):
    """Get Deribit API result."""
    attempts = retry + 1
    for attempt in range(attempts):
        start = time.time()
        try:
            response = httpx.get(
                f"{API_URL}/{method}",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(f"Deribit {method} error: {payload['error']}")
            return payload["result"]
        except HTTPX_ERRORS:
            if attempt == retry:
                raise
            time.sleep(1)
        finally:
            elapsed = time.time() - start
            if elapsed < MIN_ELAPSED_PER_REQUEST:
                time.sleep(MIN_ELAPSED_PER_REQUEST - elapsed)
    raise AssertionError("unreachable")
