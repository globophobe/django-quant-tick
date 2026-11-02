"""Public market requests with one process-wide allowance per venue."""

import json
import time
from decimal import Decimal

import httpx2

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)


def get_public_json(
    base_url: str,
    path: str,
    params: dict,
    *,
    namespace: str,
    max_requests: int = 2,
    reset_seconds: int = 1,
    retry: int = 30,
):
    total_key = f"{namespace}_TOTAL_REQUESTS"
    reset_key = f"{namespace}_MAX_REQUESTS_RESET"
    for attempt in range(retry + 1):
        throttle_api_requests(reset_key, total_key, reset_seconds, max_requests)
        increment_api_total_requests(total_key)
        try:
            response = httpx2.get(f"{base_url}{path}", params=params, timeout=30)
            response.raise_for_status()
            payload = json.loads(response.text, parse_float=Decimal)
            if isinstance(payload, dict) and str(payload.get("code")) in {
                "50011",
                "50040",
            }:
                if attempt == retry:
                    raise RuntimeError(f"{namespace} rate limit: {payload}")
                time.sleep(2)
                continue
            return payload
        except HTTPX_ERRORS as exc:
            status = (
                exc.response.status_code
                if isinstance(exc, httpx2.HTTPStatusError)
                else None
            )
            if attempt == retry or (
                status is not None and status not in (418, 429) and status < 500
            ):
                raise
            delay = 1
            if status in (418, 429):
                try:
                    delay = max(2, float(exc.response.headers.get("Retry-After", "2")))
                except ValueError:
                    delay = 2
            time.sleep(delay)
    raise AssertionError("unreachable")
