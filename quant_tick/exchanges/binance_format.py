"""Candle and individual-trade collection for keyless Binance-format market APIs."""

from datetime import datetime
from urllib.parse import parse_qsl, urlsplit

from quant_tick.controllers import ExchangeREST

from .binance.base import BinanceMixin
from .binance.candles import fetch_binance_format_candles, get_binance_interval
from .binance.trades import get_binance_trades_timestamp


def raw_trades(request, api_symbol: str, start: datetime, end: datetime, *, first_id=0):
    """Locate an older raw ID, then walk forward through every fill in [start, end).

    Aggregate trades supply only the seek anchor. Starting at its first raw ID
    preserves fills whose aggregate spans the requested lower time boundary.
    """
    cursor = seek_raw_trade(request, api_symbol, start, first_id=first_id)
    rows = []
    previous_id = cursor - 1
    previous_time = None
    while True:
        page = request(
            "/historicalTrades", {"symbol": api_symbol, "fromId": cursor, "limit": 1000}
        )
        for row in page:
            trade_id = int(row["id"])
            timestamp = get_binance_trades_timestamp(row)
            if trade_id <= previous_id or (
                previous_time is not None and timestamp < previous_time
            ):
                raise ValueError("Raw trade pagination is not strictly ordered by ID")
            previous_id, previous_time = trade_id, timestamp
            if timestamp >= end:
                return rows[::-1], True, None
            if timestamp >= start:
                rows.append(row)
        if len(page) < 1000:
            return rows[::-1], True, None
        cursor = previous_id + 1


def seek_raw_trade(request, api_symbol, start, *, first_id=0):
    timestamp = int(start.timestamp() * 1000)
    anchors = request(
        "/aggTrades",
        {
            "symbol": api_symbol,
            "startTime": max(0, timestamp - 3_600_000),
            "endTime": timestamp - 1,
            "limit": 1000,
        },
    )
    if anchors:
        if int(anchors[-1]["T"]) >= timestamp:
            raise ValueError("Aggregate trade anchor is outside the seek window")
        return int(anchors[-1]["f"])
    first = request(
        "/historicalTrades", {"symbol": api_symbol, "fromId": first_id, "limit": 1}
    )
    if not first:
        return first_id
    if int(first[0]["time"]) >= timestamp:
        return int(first[0]["id"])
    latest = request("/historicalTrades", {"symbol": api_symbol, "limit": 1})
    if int(latest[0]["time"]) < timestamp:
        return int(latest[0]["id"])
    # A quiet seek hour need not mean exhausted history. Find its raw-ID boundary
    # without downloading every trade since listing or scanning empty hours.
    low, high = int(first[0]["id"]), int(latest[0]["id"])
    while low < high:
        middle = (low + high) // 2
        page = request(
            "/historicalTrades", {"symbol": api_symbol, "fromId": middle, "limit": 1}
        )
        if page and int(page[0]["id"]) < middle:
            raise ValueError("Raw trade seek did not advance")
        if not page or int(page[0]["time"]) >= timestamp:
            high = middle
        else:
            low = int(page[0]["id"]) + 1
    return max(int(first[0]["id"]), low - 1)


def binance_format_candles(
    request, api_symbol, timestamp_from, timestamp_to, resolution="1m"
):
    def response(url, timestamp_from, pagination_id):
        params = dict(parse_qsl(urlsplit(url).query))
        params["endTime"] = pagination_id
        return request("/klines", params)[::-1]

    return fetch_binance_format_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        interval=get_binance_interval(resolution),
        get_response=response,
    )


class BinanceFormatTrades(BinanceMixin, ExchangeREST):
    partition_scoped = True
    request = None
    first_trade_id = 0

    def get_pagination_id(self, timestamp_to):
        return timestamp_to

    def iter_api(self, timestamp_from, pagination_id):
        return raw_trades(
            self.request,
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            first_id=self.first_trade_id,
        )

    def is_history_exhausted(self, timestamp_to):
        first = self.request(
            "/historicalTrades",
            {
                "symbol": self.symbol.api_symbol,
                "fromId": self.first_trade_id,
                "limit": 1,
            },
        )
        return not first or get_binance_trades_timestamp(first[0]) >= timestamp_to

    def get_candles(self, timestamp_from, timestamp_to):
        return binance_format_candles(
            self.request, self.symbol.api_symbol, timestamp_from, timestamp_to
        )
