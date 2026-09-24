import json
import time
from datetime import UTC, datetime
from decimal import Decimal

import httpx2

from quant_tick.controllers import HTTPX_ERRORS

from .constants import API_URL, MIN_ELAPSED_PER_REQUEST


def to_millis(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return int(timestamp.timestamp() * 1000)


def get_phoenix_response(path: str, params: dict, retry: int = 30):
    for attempt in range(retry + 1):
        start = time.monotonic()
        try:
            response = httpx2.get(f"{API_URL}{path}", params=params, timeout=30)
            response.raise_for_status()
            return json.loads(response.text, parse_float=Decimal)
        except HTTPX_ERRORS as exc:
            status = (
                exc.response.status_code
                if isinstance(exc, httpx2.HTTPStatusError)
                else None
            )
            if attempt == retry or (
                status is not None and status != 429 and status < 500
            ):
                raise
            delay = 1
            if status == 429:
                try:
                    delay = max(2, float(exc.response.headers.get("Retry-After", "2")))
                except ValueError:
                    delay = 2
            time.sleep(delay)
        finally:
            elapsed = time.monotonic() - start
            if elapsed < MIN_ELAPSED_PER_REQUEST:
                time.sleep(MIN_ELAPSED_PER_REQUEST - elapsed)
    raise AssertionError("unreachable")
