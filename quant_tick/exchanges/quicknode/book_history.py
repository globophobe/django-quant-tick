"""Bounded historical updates between two independently known order books.

SQL returns repeated changes to an order in reverse source occurrence order.
Validate each state transition during replay and reconcile the full terminal
order map before publishing summaries or advancing the saved checkpoint.
"""

import time

import pandas as pd

from .api import (
    BORROW_LIQUIDATION_DIRECTIONS,
    coin_literal,
    query,
    time_literal,
    timestamp,
)


def indexed_through(terminal, *, retry=30):
    height = int(terminal["block_number"])
    moment = timestamp(terminal["timestamp"])
    lower, upper = (
        time_literal(moment - pd.Timedelta(minutes=1)),
        time_literal(moment + pd.Timedelta(minutes=2)),
    )
    statements = [
        "SELECT max(block_number) AS height FROM "
        + table
        + f" WHERE block_number >= {height} AND block_number <= {height + 10000}"
        + f" AND block_time >= {lower} AND block_time < {upper}"
        for table in ("hyperliquid_order_book_diffs", "hyperliquid_trades")
    ]
    for attempt in range(retry + 1):
        rows = query(" UNION ALL ".join(statements) + " LIMIT 2")
        if len(rows) == 2 and all(int(row["height"]) >= height for row in rows):
            return
        if attempt < retry:
            time.sleep(1)
    raise ValueError("QuickNode history has not reached the order book snapshot")


def history(coin, start, terminal, *, assert_lease_owned=None):
    market = coin_literal(coin)
    first, last = int(start["block_number"]), int(terminal["block_number"])
    lower = timestamp(start["timestamp"])
    stop = timestamp(terminal["timestamp"]) + pd.Timedelta(milliseconds=1)
    if first > last or lower > stop:
        raise ValueError("Order book history boundaries are reversed")
    while lower < stop:
        if assert_lease_owned:
            assert_lease_owned()
        upper = min(lower + pd.Timedelta(minutes=1), stop)
        if upper == stop:
            indexed_through(terminal)
        begin, end = time_literal(lower), time_literal(upper)
        where = (
            f"coin={market} AND block_number > {first} AND block_number <= {last} "
            f"AND block_time >= {begin} AND block_time < {end}"
        )
        rows = query(
            "SELECT groupArray((block_number,toString(block_time),side,toString(price),"
            "oid,source_occurrence,diff_type,toString(size),toString(orig_size),toString(new_size))) AS events "
            "FROM hyperliquid_order_book_diffs FINAL WHERE " + where + " LIMIT 1"
        )
        events = rows[0]["events"] if len(rows) == 1 else None
        if not isinstance(events, list):
            raise TypeError("QuickNode returned an invalid book history window")
        trades = query(
            "SELECT block_number,toString(block_time) AS time,trade_id,side,"
            "toString(price) AS price,toString(size) AS size "
            "FROM hyperliquid_trades FINAL WHERE "
            + where
            + f" AND NOT (buyer_dir IN {BORROW_LIQUIDATION_DIRECTIONS} AND seller_dir=buyer_dir) "
            "ORDER BY block_number,trade_id LIMIT 100000"
        )
        if len(trades) == 100000:
            raise ValueError("QuickNode order book trade window reached its limit")
        groups = {}
        seen = set()
        # The adapter's occurrence direction was checked against raw blocks.
        for row in sorted(
            events, key=lambda item: (int(item[0]), int(item[4]), -int(item[5]))
        ):
            height, moment = int(row[0]), timestamp(row[1])
            identity = (height, int(row[4]), int(row[5]))
            if (
                identity in seen
                or not first < height <= last
                or not lower <= moment < upper
            ):
                raise ValueError(
                    "QuickNode book history contains duplicate or out-of-window events"
                )
            seen.add(identity)
            group = groups.setdefault(
                height, {"time": moment, "updates": [], "trades": []}
            )
            if group["time"] != moment:
                raise ValueError("QuickNode book block has inconsistent timestamps")
            group["updates"].append(row)
        seen.clear()
        for trade in trades:
            height, moment, trade_id = (
                int(trade["block_number"]),
                timestamp(trade["time"]),
                int(trade["trade_id"]),
            )
            if (
                trade_id in seen
                or not first < height <= last
                or not lower <= moment < upper
            ):
                raise ValueError(
                    "QuickNode book trades are duplicated or outside the window"
                )
            seen.add(trade_id)
            group = groups.setdefault(
                height, {"time": moment, "updates": [], "trades": []}
            )
            if group["time"] != moment:
                raise ValueError("QuickNode trades and book block timestamps disagree")
            group["trades"].append(trade)
        for height in sorted(groups):
            yield height, groups[height]
        lower = upper
