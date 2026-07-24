import logging
from datetime import UTC, datetime

from .api import get_deribit_result, to_millis
from .constants import TRADE_MAX_RESULTS

logger = logging.getLogger(__name__)


def get_deribit_trades_response(
    api_symbol: str,
    start_ms: int,
    pagination_id: dict[str, int],
) -> dict:
    params = {
        "instrument_name": str(api_symbol).strip(),
        "start_timestamp": start_ms,
        "count": TRADE_MAX_RESULTS,
        "sorting": "desc",
        **pagination_id,
    }
    return get_deribit_result("get_last_trades_by_instrument", params)


def get_deribit_trade_timestamp(trade: dict) -> datetime:
    return datetime.fromtimestamp(int(trade["timestamp"]) / 1000, tz=UTC)


def get_trades(
    api_symbol: str,
    timestamp_from: datetime,
    pagination_id: dict[str, int],
    log_format: str | None = None,
) -> tuple[list[dict], bool, dict[str, int] | None]:
    """Fetch one descending page and continue by instrument trade sequence."""
    result = get_deribit_trades_response(
        api_symbol,
        to_millis(timestamp_from),
        pagination_id,
    )
    trades = result.get("trades", [])
    if not trades:
        return [], True, None

    oldest = min(trades, key=lambda trade: int(trade["trade_seq"]))
    next_pagination_id = {"end_seq": int(oldest["trade_seq"]) - 1}
    is_last_iteration = not result.get("has_more", False)
    if log_format:
        timestamp = get_deribit_trade_timestamp(oldest)
        logger.info(log_format.format(timestamp=timestamp.isoformat()))
    return trades, is_last_iteration, next_pagination_id
