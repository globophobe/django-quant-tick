"""Public market requests with one process-wide allowance per venue.

Credit and rate limits fail immediately with redacted response details.
"""

import json
import logging
import time
from decimal import Decimal

import httpx2

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)

logger = logging.getLogger(__name__)

LIMIT_HEADERS = (
    "Date",
    "X-RateLimit-Limit",
    "X-RateLimit-Used",
    "X-RateLimit-Remaining",
    "X-RateLimit-Request-Cost",
    "X-RateLimit-Reset",
    "X-RateLimit-Quota-Allocated",
    "X-RateLimit-Quota-Remaining",
    "X-RateLimit-Quota-Overage",
    "X-ConcurrencyLimit-Limit",
    "X-ConcurrencyLimit-Remaining",
)


def limit_response_details(response, request_headers):
    """Expose bounded response diagnostics with request credentials removed."""
    secrets = set()
    for name, value in request_headers.items():
        if name.lower() in ("authorization", "x-coinapi-key", "x-api-key") and value:
            secrets.add(value)
            if name.lower() == "authorization" and " " in value:
                secrets.add(value.split(" ", 1)[1])

    def redact(value):
        for secret in sorted(secrets, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[redacted]")
        return value

    fields = {"Retry-After": response.headers.get("Retry-After", "<not provided>")}
    fields.update(
        (name, response.headers[name])
        for name in LIMIT_HEADERS
        if name in response.headers
    )
    body = redact(response.text)
    if len(body) > 2000:
        body = body[:2000] + "...[truncated]"
    return "Provider limit details: " + json.dumps(
        {
            "headers": {name: redact(value) for name, value in fields.items()},
            "body": body,
        }
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
    headers: dict[str, str] | None = None,
):
    total_key = f"{namespace}_TOTAL_REQUESTS"
    reset_key = f"{namespace}_MAX_REQUESTS_RESET"
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
            if status in (402, 429):
                details = limit_response_details(exc.response, headers or {})
                exc.add_note(details)
                logger.error(
                    "%s API credit or rate limit reached (HTTP %s); %s",
                    namespace,
                    status,
                    details,
                )
            if attempt == retry or (
                status is not None and status != 418 and status < 500
            ):
                raise
            delay = 1
            if status == 418:
                delay = min(30, 5 * 2**attempt)
                try:
                    delay = max(
                        delay, float(exc.response.headers.get("Retry-After", "0"))
                    )
                except ValueError:
                    pass
                logger.warning(
                    "%s API throttled; retrying in %g seconds", namespace, delay
                )
            time.sleep(delay)
    raise AssertionError("unreachable")
