"""Historical candles from ordered fills; validation totals use a separate query."""

from decimal import Decimal

import httpx2
import pandas as pd

from quant_tick.lib import get_current_time, iter_chunks, parse_fixed_resolution_minutes

from .api import block_range, number, query, timestamp, trade_where
from .trades import iter_trades


def validation_candles(coin, start, end):
    """Independent minute totals, splitting scan-limited queries at minute edges."""
    rows = []
    pending = list(iter_chunks(start, end, value="1d", reverse=True))
    while pending:
        lower, upper = pending.pop()
        where = trade_where(coin, lower, upper, block_range(lower, upper))
        try:
            data = query(
                "SELECT toStartOfMinute(timestamp) AS minute, sum(size) AS notional, "
                "count() AS trades FROM hyperliquid_trades FINAL "
                f"WHERE {where} GROUP BY minute ORDER BY minute LIMIT 1441"
            )
        except httpx2.HTTPStatusError as exc:
            middle = (pd.Timestamp(lower) + (upper - lower) / 2).floor("min")
            if (
                exc.response.status_code != 400
                or not any(
                    code in exc.response.text
                    for code in ("TOO_MANY_BYTES", "TOO_MANY_ROWS")
                )
                or not lower < middle < upper
            ):
                raise
            # Keep successful queries complete and ordered, without raising the
            # scan budget or counting a boundary minute in both halves.
            pending.extend(((middle, upper), (lower, middle)))
            continue
        previous = None
        for point in data:
            moment = timestamp(point["minute"])
            if (
                not lower <= moment < upper
                or moment.floor("min") != moment
                or (previous is not None and moment <= previous)
                or int(point["trades"]) <= 0
            ):
                raise ValueError("QuickNode returned invalid validation candle bounds")
            rows.append(
                {
                    "timestamp": moment,
                    "notional": number(point["notional"], positive=True),
                    "trades": int(point["trades"]),
                }
            )
            previous = moment
    return pd.DataFrame(rows, columns=["timestamp", "notional", "trades"]).set_index(
        "timestamp"
    )


def quicknode_candles(coin, timestamp_from, timestamp_to, resolution="1m"):
    minutes = parse_fixed_resolution_minutes(resolution)
    interval = pd.Timedelta(minutes=minutes)
    start = timestamp(timestamp_from).ceil(interval)
    end = min(timestamp(timestamp_to), timestamp(get_current_time())).floor(interval)
    rows = []
    for lower, upper in iter_chunks(start, end, value="1h"):
        for trade in iter_trades(coin, lower, upper):
            moment = trade["timestamp"].floor(interval)
            price, size = trade["price"], trade["size"]
            if not rows or rows[-1]["timestamp"] != moment:
                rows.append(
                    {
                        "timestamp": moment,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "notional": Decimal(0),
                        "volume": Decimal(0),
                        "trades": 0,
                    }
                )
            candle = rows[-1]
            candle["high"], candle["low"] = (
                max(candle["high"], price),
                min(candle["low"], price),
            )
            candle["close"] = price
            candle["notional"] += size
            candle["volume"] += price * size
            candle["trades"] += 1
    return pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "notional",
            "volume",
            "trades",
        ],
    ).set_index("timestamp")
