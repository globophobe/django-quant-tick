from datetime import UTC, datetime, timedelta
from functools import partial

from quant_tick.controllers import iter_api
from quant_tick.exchanges.binance.api import get_binance_api_response

from .constants import API_URL, MIN_ELAPSED_PER_REQUEST, TRADE_MAX_RESULTS

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def get_binance_futures_trade_url(
    url: str,
    timestamp_from: datetime | None = None,
    pagination_id: int | None = None,
) -> str:
    if pagination_id is not None:
        return f"{url}&fromId={pagination_id}"
    return url


def get_binance_futures_trade_pagination_id(
    timestamp: datetime,
    last_data: list | None = None,
    data: list | None = None,
) -> int | None:
    data = data or []
    if not data:
        return None
    oldest_id = int(data[-1]["a"])
    if oldest_id == 0:
        return None
    if (
        len(data) == TRADE_MAX_RESULTS
        and last_data
        and oldest_id == int(last_data[-1]["a"])
    ):
        return None
    return max(0, oldest_id - len(data))


def get_binance_futures_trade_timestamp(trade: dict) -> datetime:
    return EPOCH + timedelta(milliseconds=int(trade["T"]))


def get_binance_futures_trades(
    symbol: str,
    timestamp_from: datetime,
    pagination_id: int | None,
    *,
    log_format: str | None = None,
) -> tuple[list[dict], bool, int | None]:
    """Get Binance Futures trades."""
    url = f"{API_URL}/aggTrades?symbol={symbol}&limit={TRADE_MAX_RESULTS}"
    return iter_api(
        url,
        get_binance_futures_trade_pagination_id,
        get_binance_futures_trade_timestamp,
        partial(get_binance_api_response, get_binance_futures_trade_url),
        TRADE_MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
