"""Recognize provider throttling without retrying unrelated node failures."""

import math
import re

from ..public import RPCRateLimitError, retry_after_seconds


def check_rpc_response(response, payload):
    limits = []
    for result in payload if isinstance(payload, list) else [payload]:
        if not isinstance(result, dict):
            return
        if "error" not in result:
            continue
        error = result["error"]
        if not isinstance(error, dict):
            return
        code, message = error.get("code"), error.get("message", "")
        if not isinstance(message, str):
            return
        # Solana also uses -32007 for skipped slots and -32011 for absent history.
        if code not in (429, -32008, -32029) and not (
            code in (-32007, -32011)
            and re.search(
                r"(?:rate|requests?)[ -]limit|requests? per (?:second|minute)|"
                r"too many requests",
                message,
                re.IGNORECASE,
            )
        ):
            return
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
        limits.append((code, retry_after_seconds(response, default=delay)))
    if limits:
        code, delay = max(limits, key=lambda limit: limit[1] or 0)
        raise RPCRateLimitError(code, delay)
