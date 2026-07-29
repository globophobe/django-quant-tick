import logging
from datetime import UTC, datetime

from quant_tick.lib import get_current_time

from .api import get_deribit_result, to_millis
from .constants import (
    API_URL,
    HISTORY_API_URL,
    HISTORY_TRADE_MAX_RESULTS,
    RECENT_TRADE_WINDOW,
    TRADE_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


def get_deribit_trades_response(
    api_symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    *,
    history: bool,
) -> dict:
    params = {
        "instrument_name": str(api_symbol).strip(),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "count": HISTORY_TRADE_MAX_RESULTS if history else TRADE_MAX_RESULTS,
        "sorting": "asc",
    }
    if history:
        params["include_old"] = True
    return get_deribit_result(
        "get_last_trades_by_instrument_and_time",
        params,
        api_url=HISTORY_API_URL if history else API_URL,
    )


def get_deribit_trade_timestamp(trade: dict) -> datetime:
    return datetime.fromtimestamp(int(trade["timestamp"]) / 1000, tz=UTC)


def get_trades(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    log_format: str | None = None,
    *,
    history: bool | None = None,
) -> tuple[list[dict], bool, None]:
    """Get Deribit trades for an exact time window."""
    start_timestamp = to_millis(timestamp_from)
    end_timestamp = to_millis(timestamp_to) - 1
    if end_timestamp < start_timestamp:
        return [], True, None
    if history is None:
        history = timestamp_from < get_current_time() - RECENT_TRADE_WINDOW
    trades = _fetch_trade_window(
        api_symbol,
        start_timestamp,
        end_timestamp,
        history=history,
    )
    trades = _deduplicate_and_sort(api_symbol, trades)
    if log_format and trades:
        timestamp = get_deribit_trade_timestamp(trades[0])
        logger.info(log_format.format(timestamp=timestamp.isoformat()))
    return trades, True, None


def _fetch_trade_window(
    api_symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    *,
    history: bool,
) -> list[dict]:
    result = get_deribit_trades_response(
        api_symbol,
        start_timestamp,
        end_timestamp,
        history=history,
    )
    trades = list(result.get("trades", []))
    for trade in trades:
        timestamp = int(trade["timestamp"])
        if trade["instrument_name"] != api_symbol:
            raise ValueError(
                f"Deribit {api_symbol} returned instrument {trade['instrument_name']}"
            )
        if not start_timestamp <= timestamp <= end_timestamp:
            raise ValueError(
                f"Deribit {api_symbol} returned trade outside requested window"
            )
    if not result.get("has_more", False):
        return trades
    if not trades:
        raise ValueError(f"Deribit {api_symbol} returned has_more without trades")
    if start_timestamp == end_timestamp:
        raise ValueError(
            f"Deribit {api_symbol} exceeded the trade limit within one millisecond"
        )
    midpoint = (start_timestamp + end_timestamp) // 2
    return _fetch_trade_window(
        api_symbol,
        start_timestamp,
        midpoint,
        history=history,
    ) + _fetch_trade_window(
        api_symbol,
        midpoint + 1,
        end_timestamp,
        history=history,
    )


def _deduplicate_and_sort(api_symbol: str, trades: list[dict]) -> list[dict]:
    by_sequence = {}
    for trade in trades:
        sequence = int(trade["trade_seq"])
        existing = by_sequence.get(sequence)
        if existing is not None and existing != trade:
            raise ValueError(
                f"Deribit {api_symbol} returned conflicting trade sequence {sequence}"
            )
        by_sequence[sequence] = trade
    return [by_sequence[sequence] for sequence in sorted(by_sequence)]
