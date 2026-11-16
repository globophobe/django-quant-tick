"""Public market requests with one process-wide allowance per venue."""

import json
import logging
import math
import time
from datetime import UTC
from decimal import Decimal
from email.utils import parsedate_to_datetime

import httpx2

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)

logger = logging.getLogger(__name__)


def retry_after_seconds(response, default=2):
    value = response.headers.get("Retry-After", "")
    try:
        delay = float(value)
    except ValueError:
        try:
            timestamp = parsedate_to_datetime(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            return max(0, timestamp.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return default
    return delay if math.isfinite(delay) and delay >= 0 else default


def get_public_json(
    base_url: str,
    path: str,
    params: dict,
    *,
    namespace: str,
    max_requests: int = 2,
    reset_seconds: float = 1,
    retry: int = 30,
    headers: dict[str, str] | None = None,
):
    total_key = f"{namespace}_TOTAL_REQUESTS"
    reset_key = f"{namespace}_MAX_REQUESTS_RESET"
    limit_reported = False
    for attempt in range(retry + 1):
        throttle_api_requests(reset_key, total_key, reset_seconds, max_requests)
        increment_api_total_requests(total_key)
        try:
            kwargs = {"headers": headers} if headers is not None else {}
            response = httpx2.get(
                f"{base_url}{path}", params=params, timeout=30, **kwargs
            )
            response.raise_for_status()
            payload = json.loads(response.text, parse_float=Decimal)
            return payload
        except HTTPX_ERRORS as exc:
            status = (
                exc.response.status_code
                if isinstance(exc, httpx2.HTTPStatusError)
                else None
            )
            if status in (402, 429) and not limit_reported:
                logger.error(
                    "%s API credit or rate limit reached (HTTP %s)", namespace, status
                )
                limit_reported = True
            if attempt == retry or (
                status is not None and status not in (418, 429) and status < 500
            ):
                raise
            delay = 1
            if status in (418, 429):
                delay = retry_after_seconds(exc.response)
                logger.info("%s HTTP %s, sleeping %g seconds", namespace, status, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")
