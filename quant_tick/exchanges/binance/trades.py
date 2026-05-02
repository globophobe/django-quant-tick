from datetime import UTC, datetime, timedelta
from functools import partial

from quant_tick.constants import SymbolType
from quant_tick.controllers import iter_api

from .api import get_binance_api_response
from .constants import (
    FUTURES_API_URL,
    MIN_ELAPSED_PER_REQUEST,
    SPOT_API_URL,
    TRADE_MAX_RESULTS,
)

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def get_binance_trades_url(
    url: str, timestamp_from: datetime | None = None, pagination_id: int | None = None
) -> str:
    if pagination_id:
        return url + f"&fromId={pagination_id}"
    return url


def get_binance_trades_pagination_id(
    timestamp: datetime, last_data: list | None = None, data: list | None = None
) -> int:
    data = data or []
    # Like bybit, binance pagination feels like an IQ test.
    if len(data):
        last_trade = data[-1]
        last_id = last_trade["id"]
        pagination_id = last_id - len(data)
        # Is it the last_id? If so, stop_iteration
        if last_id == 1:
            return None
        # Is data fetched same as previous?
        if len(data) == TRADE_MAX_RESULTS and last_data and last_id == last_data[-1]["id"]:
            return None
        # Calculated pagination_id will be negative if remaining trades is
        # less than TRADE_MAX_RESULTS.
        elif pagination_id <= 0:
            return 1
        else:
            return pagination_id


def get_binance_trades_timestamp(trade: dict) -> datetime:
    return EPOCH + timedelta(milliseconds=int(trade["time"]))


def get_binance_trades_api_url(symbol_type: str) -> str:
    if symbol_type == SymbolType.PERPETUAL:
        return FUTURES_API_URL
    return SPOT_API_URL


def get_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: int,
    symbol_type: str = SymbolType.SPOT,
    log_format: str | None = None,
) -> list[dict]:
    api_url = get_binance_trades_api_url(symbol_type)
    url = f"{api_url}/historicalTrades?symbol={symbol}&limit={TRADE_MAX_RESULTS}"
    return iter_api(
        url,
        get_binance_trades_pagination_id,
        get_binance_trades_timestamp,
        partial(get_binance_api_response, get_binance_trades_url),
        TRADE_MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
