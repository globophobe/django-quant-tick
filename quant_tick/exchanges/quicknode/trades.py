"""Recover execution order without treating hashed trade IDs as a sequence.

SQL pages use the full primary key and retain a block across page boundaries.
Within a block, account position changes sometimes prove one unique order. When
those constraints leave alternatives (or do not reconcile), the native trade
block supplies the order. Every raw fill is checked against the SQL row set.
Order IDs alone are insufficient: previously placed stops can trigger later.
"""

from collections import defaultdict
from itertools import groupby, pairwise

import pandas as pd

from quant_tick.controllers import ExchangeREST

from ..hyperliquid.api import normalize_coin
from . import blocks
from .api import block_range, number, query, timestamp, trade_where

PAGE_SIZE = 10000
TRADE_COLUMNS = """block_number, timestamp, trade_id, price, size, side,
buyer_start_position, seller_start_position, buyer_address, seller_address,
buyer_crossed, seller_crossed"""


def position_order(rows):
    """Return an order only if position constraints uniquely determine it."""
    if len(rows) == 1:
        return rows
    users = defaultdict(list)
    for index, row in enumerate(rows):
        for side, sign in (("buyer", 1), ("seller", -1)):
            users[row[side + "_address"]].append(
                (index, row[side + "_start_position"], row["size"] * sign)
            )
    edges = [set() for _ in rows]
    incoming = [0] * len(rows)
    for changes in users.values():
        directions = {delta > 0 for _, _, delta in changes}
        if len(directions) != 1:
            continue
        changes.sort(key=lambda change: change[1], reverse=not next(iter(directions)))
        for (left, before, delta), (right, after, _) in pairwise(changes):
            if before + delta != after:
                return None
            if right not in edges[left]:
                edges[left].add(right)
                incoming[right] += 1
    ready = [index for index, count in enumerate(incoming) if not count]
    ordered = []
    while len(ready) == 1:
        index = ready.pop()
        ordered.append(rows[index])
        for following in edges[index]:
            incoming[following] -= 1
            if not incoming[following]:
                ready.append(following)
    if len(ordered) != len(rows):
        return None
    positions = {}
    for row in ordered:
        for side, sign in (("buyer", 1), ("seller", -1)):
            user, before = row[side + "_address"], row[side + "_start_position"]
            if user in positions and positions[user] != before:
                return None
            positions[user] = before + sign * row["size"]
    return ordered


def sql_trades(coin, start, end, bounds):
    where = trade_where(coin, start, end, bounds)
    cursor = None
    while True:
        after = ""
        if cursor is not None:
            block, trade = cursor
            after = f" AND block_number >= {block} AND (block_number, trade_id) > ({block}, {trade})"
        rows = query(
            f"SELECT {TRADE_COLUMNS} FROM hyperliquid_trades FINAL "
            f"WHERE {where}{after} ORDER BY block_number, trade_id LIMIT {PAGE_SIZE}"
        )
        if len(rows) > PAGE_SIZE:
            raise ValueError("QuickNode trade page exceeds its limit")
        for row in rows:
            block, trade_id = int(row["block_number"]), int(row["trade_id"])
            moment = timestamp(row["timestamp"])
            if (
                min(block, trade_id) < 0
                or row["side"] not in ("B", "A")
                or (row["buyer_crossed"], row["seller_crossed"])
                != ((1, 0) if row["side"] == "B" else (0, 1))
                or not start <= moment < end
                or (cursor is not None and (block, trade_id) <= cursor)
            ):
                raise ValueError(
                    "QuickNode trades are duplicated, unordered or outside the page"
                )
            cursor = block, trade_id
            yield {
                **row,
                "block_number": block,
                "trade_id": trade_id,
                "timestamp": moment,
                "price": number(row["price"], positive=True),
                "size": number(row["size"], positive=True),
                "buyer_start_position": number(row["buyer_start_position"]),
                "seller_start_position": number(row["seller_start_position"]),
            }
        if len(rows) < PAGE_SIZE:
            return


def iter_trades(coin, start, end, *, bounds=None):
    start, end = timestamp(start), timestamp(end)
    bounds = block_range(start, end) if bounds is None else bounds
    previous_time = None
    pending = []
    for _, rows in groupby(
        sql_trades(coin, start, end, bounds), lambda row: row["block_number"]
    ):
        rows = list(rows)
        times = [row["timestamp"] for row in rows]
        if previous_time is not None and min(times) < previous_time:
            raise ValueError("QuickNode execution timestamps are not monotonic")
        previous_time = max(times)
        pending.append(rows)
        if len(pending) == 100:
            yield from ordered_blocks(coin, pending)
            pending = []
    if pending:
        yield from ordered_blocks(coin, pending)


def ordered_blocks(coin, groups):
    ordered = [position_order(rows) for rows in groups]
    needed = [
        rows[0]["block_number"]
        for rows, result in zip(groups, ordered, strict=True)
        if result is None
    ]
    raw = blocks.get_trade_blocks(needed) if needed else {}
    previous = None
    for rows, result in zip(groups, ordered, strict=True):
        if result is None:
            result = blocks.order_from_block(coin, rows, raw[rows[0]["block_number"]])
        for row in result:
            if previous is not None and row["timestamp"] < previous:
                raise ValueError("QuickNode execution timestamps are not monotonic")
            previous = row["timestamp"]
            yield row


def trade_frame(rows):
    normalized = []
    for index, row in enumerate(rows):
        normalized.append(
            {
                "uid": str(row["block_number"]) + ":" + str(row["trade_id"]),
                "timestamp": row["timestamp"],
                "nanoseconds": 0,
                "price": row["price"],
                "notional": row["size"],
                "volume": row["price"] * row["size"],
                "tickRule": 1 if row["side"] == "B" else -1,
                "index": index,
                "id_trade": str(row["trade_id"]),
                "block_number": row["block_number"],
            }
        )
    return pd.DataFrame(
        normalized,
        columns=[
            "uid",
            "timestamp",
            "nanoseconds",
            "price",
            "notional",
            "volume",
            "tickRule",
            "index",
            "id_trade",
            "block_number",
        ],
    )


class QuickNodeTrades(ExchangeREST):
    partition_scoped = True

    def fetch_rest_data_frame(self, timestamp_from, timestamp_to):
        return trade_frame(
            iter_trades(
                normalize_coin(self.symbol.api_symbol), timestamp_from, timestamp_to
            )
        )
