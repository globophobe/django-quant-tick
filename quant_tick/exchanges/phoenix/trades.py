from collections.abc import Iterator
from datetime import datetime

from quant_tick.lib import parse_datetime

from .api import get_phoenix_response, to_millis
from .constants import TRADE_MAX_RESULTS


def get_trades(
    api_symbol: str, timestamp_from: datetime, timestamp_to: datetime
) -> tuple[list[dict], bool, None]:
    """Read every fill in [from, to), following the opaque cursor through ties.

    A transaction can contain multiple indistinguishable fill records. Preserve
    each occurrence; neither the signature nor timestamp identifies one fill.
    """
    rows = [
        row
        for page in _iter_trade_pages(
            api_symbol, timestamp_from, timestamp_to, limit=TRADE_MAX_RESULTS
        )
        for row in page
    ]
    return rows, True, None


def has_trades(
    api_symbol: str, timestamp_from: datetime, timestamp_to: datetime
) -> bool:
    """Check for older fills with one row, without paginating their history."""
    return bool(
        next(
            _iter_trade_pages(api_symbol, timestamp_from, timestamp_to, limit=1),
            [],
        )
    )


def _iter_trade_pages(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    limit: int,
) -> Iterator[list[dict]]:
    if timestamp_to <= timestamp_from:
        return
    symbol = api_symbol.strip().upper()
    params = {
        "startTime": to_millis(timestamp_from),
        "endTime": to_millis(timestamp_to),
        "limit": limit,
    }
    cursors = set()
    previous_timestamp = timestamp_to
    while True:
        result = get_phoenix_response(f"/v1/trades/{symbol}/fills", params)
        if not isinstance(result.get("data"), list) or not isinstance(
            result.get("hasMore"), bool
        ):
            raise TypeError("Phoenix fill page is malformed")
        for row in result["data"]:
            timestamp = parse_datetime(row["timestamp"])
            if (
                row["marketSymbol"] != symbol
                or not timestamp_from <= timestamp < timestamp_to
            ):
                raise ValueError(
                    "Phoenix fill is outside the requested market or window"
                )
            if timestamp > previous_timestamp:
                raise ValueError("Phoenix fills are not ordered newest first")
            previous_timestamp = timestamp
        cursor = result.get("nextCursor")
        if result["hasMore"]:
            if (
                not isinstance(cursor, str)
                or not cursor
                or cursor in cursors
                or not result["data"]
            ):
                raise ValueError("Phoenix fill pagination did not advance")
            cursors.add(cursor)
        yield result["data"]
        if not result["hasMore"]:
            return
        params = {**params, "cursor": cursor}
