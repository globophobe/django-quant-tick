"""Minute summaries computed before discarding ordered book changes.

Depth and spread are time-weighted across every replayed block. For the five
best prices before a block, net additions are remaining depth minus previous
depth plus aggressive executions at those same prices. This is displayed
additions minus withdrawals, not gross replenishment or inferred intent.
Partial starting minutes are omitted. Only a terminal order-map match makes
the returned summaries and boundary checkpoints eligible for persistence.
"""

from collections import defaultdict
from decimal import Decimal, localcontext

import pandas as pd

from quant_tick.constants import NUMERIC_PRECISION
from quant_tick.exchanges.quicknode.api import number, timestamp

from .book import Book
from .checkpoints import encode

MINUTE_NS = 60_000_000_000


class Summary:
    def __init__(self, book, start, end):
        self.book = book
        first = book.time if book.boundary else book.time + pd.Timedelta(nanoseconds=1)
        self.start = max(timestamp(start), first).ceil("min")
        self.end = timestamp(end).floor("min")
        self.cursor = book.time
        self.rows = {}
        self.states = {}

    def row(self, moment):
        return self.rows.setdefault(moment.floor("min"), defaultdict(Decimal))

    def advance(self, until):
        until = min(timestamp(until), self.end)
        while self.cursor < until:
            lower = self.cursor.floor("min")
            upper = min(lower + pd.Timedelta(minutes=1), until)
            if lower >= self.start:
                row = self.row(lower)
                metrics = self.book.metrics()
                nanos = upper.value - self.cursor.value
                row["coverage_ns"] += nanos
                for field, value in metrics.items():
                    row[field + "_weighted"] += value * nanos
                if "mid_open" not in row:
                    row["mid_open"] = metrics["mid"]
                    row["mid_high"] = metrics["mid"]
                    row["mid_low"] = metrics["mid"]
                row["mid_close"] = metrics["mid"]
                row["mid_high"] = max(row["mid_high"], metrics["mid"])
                row["mid_low"] = min(row["mid_low"], metrics["mid"])
            self.cursor = upper
            if upper == upper.floor("min") and (upper == self.end or upper.minute == 0):
                self.states[upper] = encode(self.book.state(upper))

    def block(self, moment, updates, trades):
        self.advance(moment)
        before = self.book.top(5)
        prices = [{price for price, _ in levels} for levels in before]
        consumed = [Decimal(0), Decimal(0)]
        values = defaultdict(Decimal)
        for trade in trades:
            side, price, size = (
                trade["side"],
                number(trade["price"], positive=True),
                number(trade["size"], positive=True),
            )
            if side not in ("B", "A"):
                raise ValueError("Invalid aggressive trade side")
            side_name = "buy" if side == "B" else "sell"
            values[side_name + "_notional"] += size
            values[side_name + "_volume"] += price * size
            consumed_side = 1 if side == "B" else 0
            if price in prices[consumed_side]:
                consumed[consumed_side] += size
        for update in updates:
            self.book.apply(update)
        self.book.top()
        if self.start <= moment < self.end:
            row = self.row(moment)
            for key, value in values.items():
                row[key] += value
            row["trade_count"] += len(trades)
            row["book_update_count"] += len(updates)
            row["block_count"] += 1
            for index, (side, letter) in enumerate((("bid", "B"), ("ask", "A"))):
                previous = sum(size for _, size in before[index])
                current = sum(
                    self.book.levels[letter][price] for price in prices[index]
                )
                row[side + "_net_addition_notional_5"] += (
                    current - previous + consumed[index]
                )
                row[side + "_executed_notional_5"] += consumed[index]

    def result(self):
        self.advance(self.end)
        rows = []
        for moment, row in sorted(self.rows.items()):
            if row["coverage_ns"] != MINUTE_NS:
                raise ValueError(
                    "Order book summary does not cover its complete minute"
                )
            values = {
                key.removesuffix("_weighted") + "_mean": value / MINUTE_NS
                for key, value in row.items()
                if key.endswith("_weighted")
            }
            row["coverage_us"] = row.pop("coverage_ns") / 1000
            values.update(
                {
                    key: value
                    for key, value in row.items()
                    if not key.endswith("_weighted")
                }
            )
            for field in (
                "buy_notional",
                "sell_notional",
                "buy_volume",
                "sell_volume",
                "trade_count",
                "book_update_count",
                "block_count",
                "bid_net_addition_notional_5",
                "ask_net_addition_notional_5",
                "bid_executed_notional_5",
                "ask_executed_notional_5",
            ):
                values.setdefault(field, Decimal(0))
            rows.append((moment, values))
        return rows, self.states


def replay(start_state, terminal_state, groups, requested_start, requested_end):
    with localcontext() as context:
        context.prec = NUMERIC_PRECISION
        book = Book(start_state, start_state["coin"])
        terminal = Book(terminal_state, book.coin)
        if terminal.time < timestamp(requested_end) or terminal.block < book.block:
            raise ValueError("Terminal book does not cover the requested end")
        summary = Summary(
            book, max(book.time, timestamp(requested_start)), requested_end
        )
        last_height, last_time = book.block, book.time
        for height, group in groups:
            moment = timestamp(group["time"])
            if (
                not last_height < height <= terminal.block
                or not last_time
                <= moment
                < terminal.time + pd.Timedelta(milliseconds=1)
            ):
                raise ValueError("Order book history is outside its anchor interval")
            summary.block(moment, group["updates"], group["trades"])
            # Include blocks with executions but no book diffs in the cursor.
            book.block, book.time = height, moment
            last_height, last_time = height, moment
        if book.orders != terminal.orders:
            raise ValueError(
                "QuickNode book history does not reconcile with the terminal snapshot"
            )
        return summary.result()
