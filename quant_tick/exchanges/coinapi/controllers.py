"""CoinAPI trades with exchange-reported taker direction.

Published files take precedence when their minute quantities pass the same
rounding tolerance as persisted trade validation. Receipt-time partitions can omit
boundary trades, so replace each incomplete minute with REST, never a concatenation
of the two transports.

Batch candles over contiguous missing ranges within each UTC day, then retain
hourly trade writes and progress. Replayed volume is retained as trade metadata;
only CoinAPI validation candles may be adjusted, and only when the original
record count and quantity both match after including those exact replays.
Native candles are never adjusted.

REST can append delayed trades with earlier exchange timestamps. Fetch complete
time windows before sorting: enlarge capped responses within unchanged bounds,
then split windows at the maximum limit. Never advance from a capped page's last
exchange timestamp, which could skip delayed trades. An unpageable timestamp fails.

Files and REST can replay a venue trade with a new CoinAPI UUID. Matching venue IDs
and timestamps are collapsed only when price, quantity and side agree, preserving
the first occurrence's order. Estimated or unknown sides cannot represent flow.
"""

from decimal import Decimal
from functools import cached_property

import pandas as pd

from quant_tick.controllers import ExchangeREST
from quant_tick.controllers.iterators import TradeDataIterator
from quant_tick.lib import (
    is_decimal_close,
    iter_missing,
    iter_timeframe,
    iter_window,
    parse_datetime,
)

from .api import get_coinapi_response, get_coinapi_symbol_type
from .candles import coinapi_candles
from .flat_files import CoinAPITradeFiles

PAGE_SIZE = 1000
MAX_PAGE_SIZE = 100000


def coinapi_trades(symbol, **kwargs):
    get_coinapi_symbol_type(symbol.api_symbol)
    CoinAPITrades(symbol, **kwargs).main()


class CoinAPITrades(ExchangeREST):
    partition_scoped = True

    @property
    def columns(self):
        return [
            *super().columns,
            "coinapi_uuid",
            "id_trade",
            "time_coinapi",
            "data_source",
            "coinapi_transport",
            "coinapi_replayed_notional",
        ]

    @property
    def api_symbol(self):
        return self.symbol.api_symbol

    def get_pagination_id(self, timestamp_to):
        return timestamp_to

    def get_candles(self, timestamp_from, timestamp_to):
        return coinapi_candles(self.api_symbol, timestamp_from, timestamp_to)

    def iter_missing_windows(self):
        yield from self.iter_missing_range(
            self.timestamp_from, self.timestamp_to, self.retry
        )

    def iter_missing_range(self, timestamp_from, timestamp_to, retry):
        iterator = TradeDataIterator(self.symbol)
        for start, end, existing in iterator.iter_days(
            timestamp_from, timestamp_to, retry=retry
        ):
            end = min(end, iterator.get_max_timestamp_to())
            if start < end:
                yield from iter_missing(
                    start, end, existing, value="1min", reverse=True
                )

    def iter_candle_windows(self):
        for start, end in self.iter_missing_windows():
            candles = self.get_candles(start, end)
            for window_from, window_to in iter_timeframe(
                start, end, value="1h", reverse=True
            ):
                yield window_from, window_to, candles

    @cached_property
    def flat_files(self):
        return CoinAPITradeFiles(self.api_symbol)

    def on_rest_data_frame(self, timestamp_from, timestamp_to, candles):
        frame = self.fetch_trade_data_frame(timestamp_from, timestamp_to, candles)
        candles = self.validation_candles(frame, candles)
        self.on_data_frame(self.symbol, timestamp_from, timestamp_to, frame, candles)
        return frame

    @staticmethod
    def validation_candles(frame, candles):
        """Reconcile vendor-counted replays without relaxing missing-trade checks."""
        if (
            frame.empty
            or "coinapi_replayed_notional" not in frame
            or "data_source" not in candles
            or "notional" not in candles
            or "trades_count" not in candles
        ):
            return candles
        quantities = frame[["notional", "coinapi_replayed_notional"]].copy()
        quantities["trades_count"] = (
            1 + quantities.coinapi_replayed_notional // quantities.notional
        ).astype("int64")
        quantities = quantities.groupby(frame.timestamp.dt.floor("min")).sum()
        adjusted = candles.copy()
        for minute, quantity, replayed, count in quantities.itertuples(name=None):
            if (
                replayed > 0
                and minute in candles.index
                and candles.loc[minute, "data_source"] == "coinapi"
                and candles.loc[minute, "notional"] > 0
                and count == candles.loc[minute, "trades_count"]
                and is_decimal_close(
                    quantity + replayed, candles.loc[minute, "notional"]
                )
            ):
                adjusted.loc[minute, "notional"] = quantity
        return adjusted

    def fetch_trade_data_frame(self, timestamp_from, timestamp_to, candles):
        if candles.empty or "notional" not in candles:
            return self.fetch_rest_data_frame(timestamp_from, timestamp_to)
        rows = self.flat_files.get_trades(timestamp_from, timestamp_to)
        if rows is None:
            return self.fetch_rest_data_frame(timestamp_from, timestamp_to)
        archive = self.get_data_frame(self.parse_data(rows[::-1]))
        candles = self.validation_candles(archive, candles)
        quantities = (
            archive.groupby(archive.timestamp.dt.floor("min")).notional.sum()
            if not archive.empty
            else {}
        )
        covered = []
        for start, end in iter_window(timestamp_from, timestamp_to, value="1min"):
            if start not in candles.index:
                continue
            quantity = quantities.get(start, Decimal(0))
            expected = candles.loc[start, "notional"]
            if bool(quantity) == bool(expected) and is_decimal_close(
                quantity, expected
            ):
                covered.append(start)
        frames = []
        if covered and not archive.empty:
            frames.append(archive.loc[archive.timestamp.dt.floor("min").isin(covered)])
        for start, end in iter_missing(
            timestamp_from, timestamp_to, covered, value="1min"
        ):
            frames.append(self.fetch_rest_data_frame(start, end))
        # Empty REST frames would coerce timezone-aware timestamps to object.
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return self.get_data_frame([])
        # File and REST frames cover disjoint minutes but can be interleaved.
        result = pd.concat(frames, ignore_index=True).sort_values(
            "timestamp", kind="stable"
        )
        result["index"] = range(len(result))
        return result

    def iter_api(self, timestamp_from, pagination_id):
        windows = [
            (pd.Timestamp(timestamp_from), pd.Timestamp(pagination_id), PAGE_SIZE)
        ]
        rows = []
        observed_ids = set()
        while windows:
            start, end, limit = windows.pop()
            if start >= end:
                continue
            page = get_coinapi_response(
                f"/v1/trades/{self.api_symbol}/history",
                {
                    "time_start": start.isoformat(),
                    "time_end": end.isoformat(),
                    "limit": limit,
                    "include_id": "true",
                },
            )
            timestamps = [parse_datetime(row["time_exchange"]) for row in page]
            if any(timestamp < start or timestamp > end for timestamp in timestamps):
                raise ValueError("CoinAPI trades are outside the requested window")
            observed_ids.update(
                row["uuid"]
                for row, timestamp in zip(page, timestamps, strict=True)
                if timestamp < pagination_id
            )
            if len(page) >= limit:
                if limit < MAX_PAGE_SIZE:
                    windows.append((start, end, min(limit * 10, MAX_PAGE_SIZE)))
                    continue
                midpoint = (start + (end - start) // 2).floor("100ns")
                if len(set(timestamps)) == 1 or not start < midpoint < end:
                    raise ValueError("CoinAPI trade timestamp exceeds the page limit")
                windows.extend(
                    ((midpoint, end, PAGE_SIZE), (start, midpoint, PAGE_SIZE))
                )
                continue
            rows.extend(
                (timestamp, row)
                for row, timestamp in zip(page, timestamps, strict=True)
                if timestamp < end
            )
        if not observed_ids <= {row["uuid"] for _, row in rows}:
            raise ValueError("CoinAPI history omitted previously observed trades")
        rows.sort(key=lambda item: item[0])
        return [row for _, row in reversed(rows)], True, None

    def parse_data(self, data):
        rows = []
        seen = set()
        venue_trades = {}
        for row in reversed(data):
            if row.get("symbol_id", self.api_symbol) != self.api_symbol:
                raise ValueError("CoinAPI trade has an unexpected symbol")
            side = row["taker_side"]
            if side not in ("BUY", "SELL"):
                raise ValueError(
                    "CoinAPI trade requires a reported BUY or SELL taker side"
                )
            uid = row["uuid"]
            if not uid or uid in seen:
                raise ValueError("CoinAPI trade UUID is missing or duplicated")
            seen.add(uid)
            timestamp = parse_datetime(row["time_exchange"])
            price = Decimal(str(row["price"]))
            quantity = Decimal(str(row["size"]))
            if (
                not price.is_finite()
                or not quantity.is_finite()
                or min(price, quantity) <= 0
            ):
                raise ValueError(
                    "CoinAPI trade price and size must be positive and finite"
                )
            trade_id = str(row["id_trade"]) if row.get("id_trade") is not None else None
            if trade_id:
                identity = trade_id, timestamp
                if identity in venue_trades:
                    previous = rows[venue_trades[identity]]
                    if (
                        previous["price"],
                        previous["notional"],
                        previous["tickRule"],
                    ) != (price, quantity, 1 if side == "BUY" else -1):
                        raise ValueError(
                            "CoinAPI exchange trade ID has conflicting values"
                        )
                    previous["coinapi_replayed_notional"] += quantity
                    continue
                venue_trades[identity] = len(rows)
            rows.append(
                {
                    "uid": uid,
                    "timestamp": timestamp,
                    "nanoseconds": timestamp.nanosecond,
                    "price": price,
                    "notional": quantity,
                    "volume": price * quantity,
                    "tickRule": 1 if side == "BUY" else -1,
                    "index": len(rows),
                    "coinapi_uuid": uid,
                    "id_trade": trade_id,
                    "time_coinapi": row["time_coinapi"],
                    "data_source": "coinapi",
                    "coinapi_transport": row.get("coinapi_transport", "rest"),
                    "coinapi_replayed_notional": Decimal(0),
                }
            )
        return rows[::-1]
