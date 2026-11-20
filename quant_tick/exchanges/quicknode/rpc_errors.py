"""Classify temporary failures for public, read-only RPC requests.

Internal errors and explicit timeouts/unavailability share the HTTP retry budget.
Permanent or unknown errors in any batch member fail with sanitized diagnostics.
"""

import math
import os
import re
from urllib.parse import unquote, urlsplit

from ..public import RPCRateLimitError, RPCRetryError, retry_after_seconds


def safe_message(message):
    for name in (
        "QUICKNODE_API_KEY",
        "QUICKNODE_SOLANA_URL",
        "QUICKNODE_HYPERCORE_URL",
    ):
        value = os.environ.get(name, "").strip()
        if not value:
            continue
        message = message.replace(value, "[redacted]")
        if name.endswith("_URL"):
            try:
                parts = urlsplit(value).path.split("/")
            except ValueError:
                continue
            for part in parts:
                if part and part not in ("evm", "info", "hypercore", "nanoreth"):
                    message = message.replace(part, "[redacted]")
                    message = message.replace(unquote(part), "[redacted]")
    message = re.sub(r"https?://[^\s<>\"']+", "[redacted URL]", message)
    return " ".join(message.split())[:1000]


def check_rpc_response(response, payload):
    failures = []
    for result in payload if isinstance(payload, list) else [payload]:
        if not isinstance(result, dict):
            return
        if "error" not in result:
            continue
        error = result["error"]
        if not isinstance(error, dict):
            return
        code, message = error.get("code"), error.get("message", "")
        if type(code) is not int or not isinstance(message, str):
            return
        # Solana also uses -32007 for skipped slots and -32011 for absent history.
        rate_limit = code in (429, -32008, -32029) or (
            code in (-32007, -32011)
            and re.search(
                r"(?:rate|requests?)[ -]limit|requests? per (?:second|minute)|"
                r"too many requests",
                message,
                re.IGNORECASE,
            )
        )
        temporary = code == -32603 or (
            -32099 <= code <= -32000
            and re.search(
                r"\b(?:timed? out|timeout|deadline exceeded|internal (?:server )?error|"
                r"(?:service|temporarily) unavailable|node (?:is )?unhealthy|"
                r"node is behind|minimum context slot (?:has )?not (?:been )?reached|"
                r"block status not available yet)\b",
                message,
                re.IGNORECASE,
            )
        )
        description = safe_message(message)
        if not rate_limit and not temporary:
            raise ValueError(f"Provider RPC error ({code}): {description}")
        delay = None
        timing = re.search(
            r"\b(?:retry|try again|wait)(?:\s+(?:after|in|for))?\s+"
            r"(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s)\b",
            message,
            re.IGNORECASE,
        )
        if timing:
            seconds = float(timing[1])
            if timing[2].lower().startswith("m"):
                seconds /= 1000
            if math.isfinite(seconds):
                delay = seconds
        delay = retry_after_seconds(response, default=delay)
        failures.append(
            RPCRateLimitError(code, delay)
            if rate_limit
            else RPCRetryError(code, description, delay)
        )
    if failures:
        failure = next(
            (item for item in failures if isinstance(item, RPCRateLimitError)),
            failures[0],
        )
        waits = [item.delay for item in failures if item.delay is not None]
        failure.delay = max(waits) if waits else None
        raise failure
