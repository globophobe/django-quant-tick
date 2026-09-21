import time
from datetime import UTC, datetime

import httpx2

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)

from .constants import (
    API_URL,
    BYBIT_MAX_REQUESTS_RESET,
    BYBIT_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    MIN_ELAPSED_PER_REQUEST,
)


def to_millis(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return int(timestamp.timestamp() * 1000)


def get_bybit_result(path: str, params: dict, retry: int = 30) -> dict:
    """Fetch within the process-wide Bybit request budget, counting every attempt.

    HTTP and ``10006`` failures share the retry budget. Rate-limit retries wait
    at least two seconds and honor a later reset timestamp.
    """
    attempts = retry + 1
    for attempt in range(attempts):
        throttle_api_requests(
            BYBIT_MAX_REQUESTS_RESET,
            BYBIT_TOTAL_REQUESTS,
            MAX_REQUESTS_RESET,
            MAX_REQUESTS,
        )
        start = time.time()
        try:
            increment_api_total_requests(BYBIT_TOTAL_REQUESTS)
            response = httpx2.get(
                f"{API_URL}{path}",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("retCode") == 10006 and attempt < retry:
                try:
                    reset_ms = int(response.headers["X-Bapi-Limit-Reset-Timestamp"])
                    delay = max(2.0, reset_ms / 1000 - time.time())
                except (KeyError, ValueError):
                    delay = 2.0
                time.sleep(delay)
                continue
            if payload.get("retCode") != 0:
                raise RuntimeError(
                    f"Bybit {path} error {payload.get('retCode')}: "
                    f"{payload.get('retMsg')}"
                )
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
