"""Spot trade ingestion with exchange-reported taker direction.

History is ascending and paginated by time. Re-fetch the last timestamp of a
full page, retaining only earlier rows, so tied trades cannot fall between pages.
If a timestamp fills the API's maximum page, fail before persisting that slice.
Estimated or unknown sides are rejected; they cannot stand in for reported flow.
"""

from decimal import Decimal

from quant_tick.controllers import ExchangeREST
from quant_tick.lib import parse_datetime

from .api import get_coinapi_response, require_spot_symbol
from .candles import coinapi_candles

PAGE_SIZE = 1000
MAX_PAGE_SIZE = 100000


def coinapi_trades(symbol, **kwargs):
    require_spot_symbol(symbol.api_symbol)
    CoinAPITrades(symbol, **kwargs).main()


class CoinAPITrades(ExchangeREST):
    partition_scoped = True

    @property
    def columns(self):
        return [*super().columns, "coinapi_uuid", "id_trade", "time_coinapi"]

    def get_pagination_id(self, timestamp_to):
        return timestamp_to

    def get_candles(self, timestamp_from, timestamp_to):
        return coinapi_candles(self.symbol.api_symbol, timestamp_from, timestamp_to)

    def iter_api(self, timestamp_from, pagination_id):
        cursor = timestamp_from
        time_start = cursor.isoformat()
        rows = []
        limit = PAGE_SIZE
        boundary_ids = set()
        while cursor < pagination_id:
            page = get_coinapi_response(
                f"/v1/trades/{self.symbol.api_symbol}/history",
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
            if row.get("symbol_id", self.symbol.api_symbol) != self.symbol.api_symbol:
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
                }
            )
        return rows
