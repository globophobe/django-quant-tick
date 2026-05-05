import json
import time
from datetime import UTC, datetime
from decimal import Decimal

import httpx

from quant_tick.controllers import HTTPX_ERRORS

from .constants import API_URL, INFO_PATH, MIN_ELAPSED_PER_REQUEST


def to_millis(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return int(timestamp.timestamp() * 1000)


def normalize_coin(api_symbol: str) -> str:
    raw = str(api_symbol).strip()
    if raw.startswith("Crypto."):
        raw = raw.removeprefix("Crypto.")
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    return raw


def post_hyperliquid_info(payload: dict, retry: int = 30) -> list[dict]:
    start = time.time()
    try:
        response = httpx.post(
            f"{API_URL}{INFO_PATH}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code == 200:
            return json.loads(response.text, parse_float=Decimal)
        response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            return post_hyperliquid_info(payload, retry=retry - 1)
        raise
    finally:
        elapsed = time.time() - start
        if elapsed < MIN_ELAPSED_PER_REQUEST:
            time.sleep(MIN_ELAPSED_PER_REQUEST - elapsed)
