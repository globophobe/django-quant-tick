"""Public market requests with rate limiting and Retry-After handling."""

import json
import logging
import math
import time
from collections.abc import Callable
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


class RPCRetryError(RuntimeError):
    """A temporary RPC failure with a safe message and optional wait in seconds."""

    def __init__(self, code, message, delay=None):
        super().__init__(f"Provider RPC error ({code}): {message}")
        self.code = code
        self.delay = delay


class RPCRateLimitError(RPCRetryError):
    """A provider throttle with an optional retry delay in seconds."""

    def __init__(self, code, delay):
        super().__init__(code, "rate limit reached", delay)


def retry_after_seconds(response, default=1):
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
    max_requests: int | None = 2,
    reset_seconds: float = 1,
    retry: int = 30,
    headers: dict[str, str] | None = None,
    json_body: dict | list | None = None,
    redact_url: bool = False,
    check_response: Callable[[httpx2.Response, object], None] | None = None,
):
    """Use response quotas when max_requests is None; share HTTP/RPC retries.

    Honor provider retry timing; otherwise back off from one to thirty seconds
    for rate limits. Timeouts, transport and server errors use a one-second wait.
    """
    total_key = f"{namespace}_TOTAL_REQUESTS"
    reset_key = f"{namespace}_MAX_REQUESTS_RESET"
    limit_reported = False
    for attempt in range(retry + 1):
        if max_requests is not None:
            throttle_api_requests(reset_key, total_key, reset_seconds, max_requests)
            increment_api_total_requests(total_key)
        try:
            kwargs = {"headers": headers} if headers is not None else {}
            if redact_url:
                # Endpoint credentials live in the URL. The transport avoids the
                # client's URL log; errors carry only a credential-free request.
                safe_request = httpx2.Request("POST", "https://quiknode.pro/rpc")
                try:
                    request = httpx2.Request(
                        "POST",
                        f"{base_url}{path}",
                        json=json_body,
                        **kwargs,
                        extensions={
                            "timeout": {
                                key: 30 for key in ("connect", "read", "write", "pool")
                            }
                        },
                    )
                    with httpx2.HTTPTransport() as transport:
                        response = transport.handle_request(request)
                        response.request = safe_request
                        response.read()
                    logger.info(
                        'HTTP Request: %s %s "%s %d %s"',
                        safe_request.method,
                        safe_request.url,
                        response.http_version,
                        response.status_code,
                        response.reason_phrase,
                    )
                except httpx2.RequestError:
                    raise httpx2.ConnectError(
                        "Provider endpoint transport failed", request=safe_request
                    ) from None
            elif json_body is None:
                response = httpx2.get(
                    f"{base_url}{path}", params=params, timeout=30, **kwargs
                )
            else:
                response = httpx2.post(
                    f"{base_url}{path}", json=json_body, timeout=30, **kwargs
                )
            response.raise_for_status()
            payload = json.loads(response.text, parse_float=Decimal)
            if check_response is not None:
                check_response(response, payload)
            if max_requests is None:
                limit = response.headers.get("X-RateLimit-Limit")
                remaining = response.headers.get("X-RateLimit-Remaining")
                if limit is not None and remaining is not None:
                    rate = int(limit)
                    if rate > 0 and int(remaining) <= 0:
                        # The limit is the refill rate, not the burst capacity.
                        delay = 1 / rate
                        logger.info(
                            "%s quota exhausted, sleeping %g seconds", namespace, delay
                        )
                        time.sleep(delay)
            return payload
        except (*HTTPX_ERRORS, RPCRetryError) as exc:
            rpc_limit = isinstance(exc, RPCRateLimitError)
            status = (
                exc.response.status_code
                if isinstance(exc, httpx2.HTTPStatusError)
                else None
            )
            if (rpc_limit or status in (402, 429)) and not limit_reported:
                logger.error(
                    "%s API credit or rate limit reached (%s %s)",
                    namespace,
                    "RPC" if rpc_limit else "HTTP",
                    exc.code if rpc_limit else status,
                )
                limit_reported = True
            if attempt == retry or (
                status is not None and status not in (408, 418, 429) and status < 500
            ):
                raise
            delay = 1
            if isinstance(exc, RPCRetryError):
                fallback = min(2**attempt, 30) if rpc_limit else 1
                delay = exc.delay if exc.delay is not None else fallback
                logger.info(
                    "%s RPC %s, sleeping %g seconds", namespace, exc.code, delay
                )
            elif status in (418, 429):
                delay = retry_after_seconds(exc.response, default=min(2**attempt, 30))
                logger.info("%s HTTP %s, sleeping %g seconds", namespace, status, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")
