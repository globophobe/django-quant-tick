import logging
import time
from datetime import datetime
from urllib.parse import urlencode

import httpx

from quant_tick.controllers import HTTPX_ERRORS
from quant_tick.lib import parse_datetime

from .constants import (
    API_URL,
    MIN_ELAPSED_PER_REQUEST,
    TRADE_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


class CoinbaseAdvancedPaginationError(RuntimeError):
    """Raised when Coinbase Advanced REST cannot prove trade coverage."""


def format_coinbase_advanced_timestamp(timestamp: datetime) -> int:
    """Coinbase Advanced market trades accepts integer Unix seconds."""
    return int(timestamp.timestamp())


def get_coinbase_advanced_trades_timestamp(trade: dict) -> datetime:
    return parse_datetime(trade["time"])


def get_coinbase_advanced_trades_url(
    symbol: str,
    timestamp_from: int,
    timestamp_to: int,
) -> str:
    query = urlencode(
        {
            "limit": TRADE_MAX_RESULTS,
            "start": timestamp_from,
            "end": timestamp_to,
        }
    )
    return f"{API_URL}/products/{symbol}/ticker?{query}"


def fetch_coinbase_advanced_trades(
    symbol: str,
    timestamp_from: int,
    timestamp_to: int,
    retry: int = 30,
) -> list[dict]:
    start = time.time()
    try:
        url = get_coinbase_advanced_trades_url(symbol, timestamp_from, timestamp_to)
        response = httpx.get(url)
        if response.status_code == 200:
            trades = response.json().get("trades", [])
            return sorted(
                trades,
                key=get_coinbase_advanced_trades_timestamp,
                reverse=True,
            )
        response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            return fetch_coinbase_advanced_trades(
                symbol,
                timestamp_from,
                timestamp_to,
                retry=retry - 1,
            )
        raise
    finally:
        elapsed = time.time() - start
        if elapsed < MIN_ELAPSED_PER_REQUEST:
            time.sleep(MIN_ELAPSED_PER_REQUEST - elapsed)


def get_coinbase_advanced_trades(
    symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    log_format: str | None = None,
) -> list[dict]:
    start = format_coinbase_advanced_timestamp(timestamp_from)
    end = format_coinbase_advanced_timestamp(timestamp_to)
    return get_coinbase_advanced_trades_window(symbol, start, end, log_format=log_format)


def get_coinbase_advanced_trades_window(
    symbol: str,
    timestamp_from: int,
    timestamp_to: int,
    log_format: str | None = None,
) -> list[dict]:
    if timestamp_from >= timestamp_to:
        return []

    trades = fetch_coinbase_advanced_trades(symbol, timestamp_from, timestamp_to)
    if trades and log_format:
        timestamp = get_coinbase_advanced_trades_timestamp(trades[-1])
        logger.info(log_format.format(timestamp=timestamp.isoformat()))
    if len(trades) < TRADE_MAX_RESULTS:
        return trades
    if timestamp_to - timestamp_from <= 1:
        raise CoinbaseAdvancedPaginationError(
            f"Coinbase Advanced returned {TRADE_MAX_RESULTS} trades for "
            f"{symbol} in one second: {timestamp_from} to {timestamp_to}."
        )

    mid = timestamp_from + ((timestamp_to - timestamp_from) // 2)
    newer = get_coinbase_advanced_trades_window(
        symbol,
        mid,
        timestamp_to,
        log_format=log_format,
    )
    older = get_coinbase_advanced_trades_window(
        symbol,
        timestamp_from,
        mid,
        log_format=log_format,
    )
    return newer + older
