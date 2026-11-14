"""Spot and Hyperliquid perpetual trades with exchange-reported taker direction.

Published files take precedence when their minute quantities match validation
candles exactly. Receipt-time partitions can omit boundary trades, so replace
each incomplete minute with REST, never a concatenation of the two transports.

REST history is ascending and paginated by time. Re-fetch the last timestamp of a
full page, retaining only earlier rows, so tied trades cannot fall between pages.
If a timestamp fills the API's maximum page, fail before persisting that slice.
Estimated or unknown sides are rejected; they cannot stand in for reported flow.
"""

from decimal import Decimal
from functools import cached_property

import pandas as pd

from quant_tick.controllers import ExchangeREST
from quant_tick.lib import iter_missing, iter_window, parse_datetime

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
        ]

    @property
    def api_symbol(self):
        return self.symbol.api_symbol

    def get_pagination_id(self, timestamp_to):
        return timestamp_to

    def get_candles(self, timestamp_from, timestamp_to):
        return coinapi_candles(self.api_symbol, timestamp_from, timestamp_to)

    @cached_property
    def flat_files(self):
        return CoinAPITradeFiles(self.api_symbol)

    def on_rest_data_frame(self, timestamp_from, timestamp_to, candles):
        frame = self.fetch_trade_data_frame(timestamp_from, timestamp_to, candles)
        self.on_data_frame(self.symbol, timestamp_from, timestamp_to, frame, candles)
        return frame

    def fetch_trade_data_frame(self, timestamp_from, timestamp_to, candles):
        if candles.empty or "notional" not in candles:
            return self.fetch_rest_data_frame(timestamp_from, timestamp_to)
        rows = self.flat_files.get_trades(timestamp_from, timestamp_to)
        if rows is None:
            return self.fetch_rest_data_frame(timestamp_from, timestamp_to)
        archive = self.get_data_frame(self.parse_data(rows[::-1]))
        quantities = (
            archive.groupby(archive.timestamp.dt.floor("min")).notional.sum()
            if not archive.empty
            else {}
        )
        covered = []
        for start, end in iter_window(timestamp_from, timestamp_to, value="1min"):
            if (
                start in candles.index
                and quantities.get(start, Decimal(0)) == candles.loc[start, "notional"]
            ):
                covered.append(start)
        frames = []
        if covered and not archive.empty:
            frames.append(archive.loc[archive.timestamp.dt.floor("min").isin(covered)])
        for start, end in iter_missing(
            timestamp_from, timestamp_to, covered, value="1min"
        ):
            frames.append(self.fetch_rest_data_frame(start, end))
        if not frames:
            return self.get_data_frame([])
        # File and REST frames cover disjoint minutes but can be interleaved.
        result = pd.concat(frames, ignore_index=True).sort_values(
            "timestamp", kind="stable"
        )
        result["index"] = range(len(result))
        return result

    def iter_api(self, timestamp_from, pagination_id):
        cursor = timestamp_from
        time_start = cursor.isoformat()
        rows = []
        limit = PAGE_SIZE
        boundary_ids = set()
        while cursor < pagination_id:
            page = get_coinapi_response(
                f"/v1/trades/{self.api_symbol}/history",
                {
                    "time_start": time_start,
                    "time_end": pagination_id.isoformat(),
                    "limit": limit,
                    "include_id": "true",
                },
            )
            timestamps = [parse_datetime(row["time_exchange"]) for row in page]
            previous = cursor
            for timestamp in timestamps:
                if timestamp < previous or timestamp > pagination_id:
                    raise ValueError("CoinAPI trades are unordered or outside the page")
                previous = timestamp
            cursor_ids = {
                row["uuid"]
                for row, timestamp in zip(page, timestamps, strict=True)
                if timestamp == cursor
            }
            if not boundary_ids <= cursor_ids:
                raise ValueError(
                    "CoinAPI page omitted previously observed boundary trades"
                )
            if len(page) < limit:
                rows.extend(
                    row
                    for row, timestamp in zip(page, timestamps, strict=True)
                    if timestamp < pagination_id
                )
                break
            boundary = timestamps[-1]
            boundary_ids = {
                row["uuid"]
                for row, timestamp in zip(page, timestamps, strict=True)
                if timestamp == boundary
            }
            if boundary == cursor:
                if limit == MAX_PAGE_SIZE:
                    raise ValueError("CoinAPI trade timestamp exceeds the page limit")
                limit = min(limit * 10, MAX_PAGE_SIZE)
                continue
            rows.extend(
                row
                for row, timestamp in zip(page, timestamps, strict=True)
                if timestamp < boundary
            )
            cursor = boundary
            time_start = page[-1]["time_exchange"]
            limit = PAGE_SIZE
        return rows[::-1], True, None

    def parse_data(self, data):
        rows = []
        seen = set()
        for index, row in enumerate(data):
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
            rows.append(
                {
                    "uid": uid,
                    "timestamp": timestamp,
                    "nanoseconds": timestamp.nanosecond,
                    "price": price,
                    "notional": quantity,
                    "volume": price * quantity,
                    "tickRule": 1 if side == "BUY" else -1,
                    "index": len(data) - index - 1,
                    "coinapi_uuid": uid,
                    "id_trade": str(row["id_trade"])
                    if row.get("id_trade") is not None
                    else None,
                    "time_coinapi": row["time_coinapi"],
                    "data_source": "coinapi",
                    "coinapi_transport": row.get("coinapi_transport", "rest"),
                }
            )
        return rows
