"""Per-order state for aggregate depth, without queue-priority reconstruction.

A checkpoint is the state after its block and before its timestamp boundary.
A live seed instead uses the timestamp of its inclusive snapshot block.
New updates may replace an order; removals and updates require prior state.
"""

from bisect import bisect_left, insort
from collections import Counter
from decimal import Decimal

import pandas as pd

from quant_tick.exchanges.quicknode.api import number, timestamp

VERSION = 1


def decimal_string(value):
    return format(value.normalize(), "f")


class Book:
    def __init__(self, state, coin):
        if state.get("version") != VERSION or state.get("coin") != coin:
            raise ValueError("Order book checkpoint format or market mismatch")
        self.coin = coin
        self.block = int(state["block_number"])
        self.time = timestamp(state["timestamp"])
        self.boundary = state["boundary"]
        if not isinstance(self.boundary, bool):
            raise TypeError("Invalid order book checkpoint boundary")
        self.orders = {}
        self.levels = {"B": Counter(), "A": Counter()}
        if self.block <= 0:
            raise ValueError("Invalid order book checkpoint block")
        for oid, side, price, size in state["orders"]:
            oid, price, size = int(oid), number(price, positive=True), number(size)
            if oid <= 0 or oid in self.orders or side not in self.levels or size < 0:
                raise ValueError("Invalid or duplicate checkpoint order")
            self.orders[oid] = (side, price, size)
            self.levels[side][price] += size
        self.prices = {
            side: sorted(price for price, size in levels.items() if size > 0)
            for side, levels in self.levels.items()
        }
        self.top()

    def change_level(self, side, price, delta):
        old = self.levels[side][price]
        new = old + delta
        if new < 0:
            raise ValueError("Order book level became negative")
        self.levels[side][price] = new
        if old == 0 and new > 0:
            insort(self.prices[side], price)
        elif old > 0 and new == 0:
            self.prices[side].pop(bisect_left(self.prices[side], price))
            del self.levels[side][price]

    def apply(self, row):
        block, moment, side, price, oid, _occurrence, kind, size, old, new = row
        block, oid = int(block), int(oid)
        moment, price = timestamp(moment), number(price, positive=True)
        if (
            block < self.block
            or moment < self.time
            or side not in self.levels
            or oid <= 0
        ):
            raise ValueError("Order book updates are out of order or invalid")
        previous = self.orders.get(oid)
        if kind == "new":
            qty = number(size)
        elif kind == "update":
            if previous != (side, price, number(old)):
                raise ValueError("Order book update does not match the resting order")
            qty = number(new)
        elif kind == "remove":
            if previous is None or previous[:2] != (side, price):
                raise ValueError("Order book removal has no matching resting order")
            qty = None
        else:
            raise ValueError("Unknown order book update type")
        if qty is not None and qty < 0:
            raise ValueError("Order book size is negative")
        if previous:
            self.change_level(previous[0], previous[1], -previous[2])
        if qty is None:
            del self.orders[oid]
        else:
            self.orders[oid] = side, price, qty
            self.change_level(side, price, qty)
        self.block, self.time = block, moment

    def top(self, count=20):
        bids = [
            (price, self.levels["B"][price])
            for price in reversed(self.prices["B"][-count:])
        ]
        asks = [(price, self.levels["A"][price]) for price in self.prices["A"][:count]]
        if not bids or not asks or bids[0][0] >= asks[0][0]:
            raise ValueError("Order book is empty or crossed")
        return bids, asks

    def state(self, boundary=None):
        is_boundary = boundary is not None
        boundary = self.time if boundary is None else timestamp(boundary)
        if boundary < self.time:
            raise ValueError("Checkpoint boundary predates processed book updates")
        return {
            "version": VERSION,
            "coin": self.coin,
            "block_number": self.block,
            "timestamp": boundary.isoformat(),
            "boundary": is_boundary,
            "orders": [
                [oid, side, decimal_string(price), decimal_string(size)]
                for oid, (side, price, size) in sorted(self.orders.items())
            ],
        }

    def metrics(self):
        bids, asks = self.top()
        mid = (bids[0][0] + asks[0][0]) / 2
        values = {"mid": mid, "spread_bps": (asks[0][0] - bids[0][0]) / mid * 10000}
        for side, levels in (("bid", bids), ("ask", asks)):
            for count in (5, 20):
                values[f"{side}_notional_{count}"] = sum(
                    (size for _, size in levels[:count]), Decimal(0)
                )
                values[f"{side}_volume_{count}"] = sum(
                    (price * size for price, size in levels[:count]), Decimal(0)
                )
        return values


def from_milliseconds(value):
    return pd.Timestamp(int(value), unit="ms", tz="UTC")
